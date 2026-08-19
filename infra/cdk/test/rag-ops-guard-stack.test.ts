import { App, aws_lambda as lambda } from 'aws-cdk-lib';
import { Match, Template } from 'aws-cdk-lib/assertions';
import { RagOpsGuardStack } from '../lib/rag-ops-guard-stack';

function inlineCode(): lambda.Code {
  return lambda.Code.fromInline('def handler(event, context):\n    return {"statusCode": 200}\n');
}

describe('RagOpsGuardStack', () => {
  test('local target uses the real application handlers and Python 3.13 contract', () => {
    const app = new App();
    const stack = new RagOpsGuardStack(app, 'LocalTestStack', {
      target: 'local',
      pythonVersion: '3.13',
      lambdaCode: inlineCode(),
    });
    const template = Template.fromStack(stack);

    template.resourcePropertiesCountIs(
      'AWS::Lambda::Function',
      {
        Runtime: 'python3.13',
        Environment: {
          Variables: Match.objectLike({
            APP_ENV: 'local',
            AWS_ENDPOINT_URL: 'http://floci:4566',
            TENANT_TABLE: Match.anyValue(),
            S3_VECTOR_BUCKET: 'rag-ops-guard-vectors-local',
            S3_VECTOR_INDEX: 'ops-knowledge-openvino-v1',
            EMBEDDING_BASE_URL: 'http://rag-ops-ovms-rag:8000/v3',
            RERANKER_BASE_URL: 'http://rag-ops-ovms-rag:8000/v3',
          }),
        },
      },
      2,
    );
    template.hasResourceProperties('AWS::Lambda::Function', {
      Handler: 'rag_ops_guard.handlers.ingest.handler',
    });
    template.hasResourceProperties('AWS::Lambda::Function', {
      Handler: 'rag_ops_guard.handlers.query.handler',
    });
  });

  test('creates one S3 Vectors bucket and one structural index per tenant', () => {
    const app = new App();
    const stack = new RagOpsGuardStack(app, 'VectorTestStack', {
      target: 'local',
      pythonVersion: '3.13',
      lambdaCode: inlineCode(),
      tenantIds: ['tenant-a', 'tenant-b'],
    });
    const template = Template.fromStack(stack);

    template.resourceCountIs('AWS::S3Vectors::VectorBucket', 1);
    template.resourceCountIs('AWS::S3Vectors::Index', 2);
    for (const tenantId of ['tenant-a', 'tenant-b']) {
      template.hasResourceProperties('AWS::S3Vectors::Index', {
        DataType: 'float32',
        Dimension: 1024,
        DistanceMetric: 'cosine',
        IndexName: `ops-knowledge-openvino-v1--${tenantId}`,
        VectorBucketName: 'rag-ops-guard-vectors-local',
      });
    }
  });

  test('creates retained pay-per-request config and tenant credential tables', () => {
    const app = new App();
    const stack = new RagOpsGuardStack(app, 'ConfigTestStack', {
      target: 'local',
      pythonVersion: '3.13',
      lambdaCode: inlineCode(),
    });
    const template = Template.fromStack(stack);

    template.resourceCountIs('AWS::DynamoDB::Table', 2);
    template.hasResourceProperties('AWS::DynamoDB::Table', {
      TableName: 'rag-ops-config',
      BillingMode: 'PAY_PER_REQUEST',
      PointInTimeRecoverySpecification: { PointInTimeRecoveryEnabled: true },
      KeySchema: [
        { AttributeName: 'PK', KeyType: 'HASH' },
        { AttributeName: 'SK', KeyType: 'RANGE' },
      ],
    });
    template.hasResourceProperties('AWS::DynamoDB::Table', {
      TableName: 'rag-ops-tenants',
      BillingMode: 'PAY_PER_REQUEST',
      PointInTimeRecoverySpecification: { PointInTimeRecoveryEnabled: true },
      KeySchema: [{ AttributeName: 'key_id', KeyType: 'HASH' }],
    });
  });

  test('Phase 5 creates a retained KMS key with a stable secret-management alias', () => {
    const app = new App();
    const stack = new RagOpsGuardStack(app, 'Phase5KmsStack', {
      target: 'local',
      pythonVersion: '3.13',
      lambdaCode: inlineCode(),
    });
    const template = Template.fromStack(stack);

    template.resourceCountIs('AWS::KMS::Key', 1);
    template.hasResourceProperties('AWS::KMS::Alias', {
      AliasName: 'alias/rag-ops-guard-config-secrets',
    });
  });

  test('tenant config admin roles enforce config and encrypted-secret LeadingKeys', () => {
    const app = new App();
    const stack = new RagOpsGuardStack(app, 'TenantIamTestStack', {
      target: 'local',
      pythonVersion: '3.13',
      lambdaCode: inlineCode(),
      tenantIds: ['tenant-a', 'tenant-b'],
    });
    const template = Template.fromStack(stack);
    const policies = JSON.stringify(template.findResources('AWS::IAM::Policy'));

    expect(policies).toContain('dynamodb:LeadingKeys');
    expect(policies).toContain('TENANT#tenant-a');
    expect(policies).toContain('TENANT#tenant-b');
    expect(policies).toContain('SECRET_SCOPE#TENANT#tenant-a');
    expect(policies).toContain('SECRET_SCOPE#TENANT#tenant-b');
    expect(policies).toContain('ForAllValues:StringEquals');
    expect(policies).toContain('kms:Decrypt');
    expect(policies).toContain('kms:GenerateDataKey');
    expect(policies).not.toContain('dynamodb:Scan');
  });

  test('AWS target is fail-closed on external HTTPS inference endpoints', () => {
    const app = new App();
    expect(
      () =>
        new RagOpsGuardStack(app, 'InvalidAwsStack', {
          target: 'aws',
          pythonVersion: '3.13',
          lambdaCode: inlineCode(),
        }),
    ).toThrow(/RAG_OPS_AWS_LLM_BASE_URL/);
  });

  test('AWS target declares real endpoint semantics and Phase 5 dashboard', () => {
    const app = new App();
    const stack = new RagOpsGuardStack(app, 'AwsTestStack', {
      target: 'aws',
      pythonVersion: '3.13',
      lambdaCode: inlineCode(),
      llmBaseUrl: 'https://llm.example.test/v1',
      embeddingBaseUrl: 'https://embedding.example.test/v1',
      rerankerBaseUrl: 'https://reranker.example.test/v1',
    });
    const template = Template.fromStack(stack);

    template.resourcePropertiesCountIs(
      'AWS::Lambda::Function',
      {
        Runtime: 'python3.13',
        Environment: {
          Variables: Match.objectLike({
            APP_ENV: 'aws',
            AWS_ENDPOINT_URL: '',
            CONFIG_TABLE: Match.anyValue(),
            TENANT_TABLE: Match.anyValue(),
            LLM_BASE_URL: 'https://llm.example.test/v1',
          }),
        },
      },
      2,
    );
    template.hasResourceProperties('AWS::CloudWatch::Dashboard', {
      DashboardName: 'rag-ops-guard-config-admin',
    });
    const dashboards = JSON.stringify(template.findResources('AWS::CloudWatch::Dashboard'));
    expect(dashboards).toContain('ConfigResolveLatencyMs');
    expect(dashboards).toContain('ConfigCacheHit');
    expect(dashboards).toContain('ConfigDbUnavailable');
    expect(dashboards).toContain('SecretDecryptFailure');
    expect(dashboards).toContain('TenantIsolationViolation');
  });

  test('local target does not create CloudWatch dashboard resources', () => {
    const app = new App();
    const stack = new RagOpsGuardStack(app, 'LocalDashboardStack', {
      target: 'local',
      pythonVersion: '3.13',
      lambdaCode: inlineCode(),
    });
    Template.fromStack(stack).resourceCountIs('AWS::CloudWatch::Dashboard', 0);
  });

  test('config table policy never grants Scan', () => {
    const app = new App();
    const stack = new RagOpsGuardStack(app, 'PolicyTestStack', {
      target: 'local',
      pythonVersion: '3.13',
      lambdaCode: inlineCode(),
    });
    const template = Template.fromStack(stack);
    const policies = template.findResources('AWS::IAM::Policy');
    const serialized = JSON.stringify(policies);
    expect(serialized).toContain('dynamodb:GetItem');
    expect(serialized).toContain('dynamodb:Query');
    expect(serialized).not.toContain('dynamodb:Scan');
  });

  test('creates HTTP API routes', () => {
    const app = new App();
    const stack = new RagOpsGuardStack(app, 'ApiTestStack', {
      target: 'local',
      pythonVersion: '3.13',
      lambdaCode: inlineCode(),
    });
    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::ApiGatewayV2::Route', {
      RouteKey: 'POST /v1/query',
    });
    template.hasResourceProperties('AWS::ApiGatewayV2::Route', {
      RouteKey: 'POST /v1/ingest',
    });
  });
});
