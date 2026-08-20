import { App, aws_lambda as lambda } from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import { RagOpsGuardStack } from '../lib/rag-ops-guard-stack';

function inlineCode(): lambda.Code {
  return lambda.Code.fromInline('def handler(event, context):\n    return {"statusCode": 200}\n');
}

function localStack(id = 'LocalTestStack', tenantIds?: string[]): Template {
  const app = new App();
  return Template.fromStack(new RagOpsGuardStack(app, id, {
    target: 'local', pythonVersion: '3.13', lambdaCode: inlineCode(), tenantIds,
  }));
}

describe('RagOpsGuardStack', () => {
  test('Phase 6 Lambdas use Python 3.13 handlers with zero custom environment variables', () => {
    const template = localStack();
    const functions = template.findResources('AWS::Lambda::Function');
    expect(Object.values(functions)).toHaveLength(2);
    for (const resource of Object.values(functions) as Array<{ Properties?: Record<string, unknown> }>) {
      expect(resource.Properties?.Runtime).toBe('python3.13');
      expect(resource.Properties).not.toHaveProperty('Environment');
    }
    template.hasResourceProperties('AWS::Lambda::Function', { Handler: 'rag_ops_guard.handlers.ingest.handler' });
    template.hasResourceProperties('AWS::Lambda::Function', { Handler: 'rag_ops_guard.handlers.query.handler' });
  });

  test('creates one structural vector index per tenant', () => {
    const template = localStack('VectorTestStack', ['tenant-a', 'tenant-b']);
    template.resourceCountIs('AWS::S3Vectors::VectorBucket', 1);
    template.resourceCountIs('AWS::S3Vectors::Index', 2);
    for (const tenantId of ['tenant-a', 'tenant-b']) {
      template.hasResourceProperties('AWS::S3Vectors::Index', {
        DataType: 'float32', Dimension: 1024, DistanceMetric: 'cosine',
        IndexName: `ops-knowledge-openvino-v1--${tenantId}`,
        VectorBucketName: 'rag-ops-guard-vectors-local',
      });
    }
  });

  test('keeps config and tenant credential tables and never stores API plaintext by schema', () => {
    const template = localStack('ConfigTestStack');
    template.resourceCountIs('AWS::DynamoDB::Table', 2);
    template.hasResourceProperties('AWS::DynamoDB::Table', {
      TableName: 'rag-ops-config', BillingMode: 'PAY_PER_REQUEST',
      KeySchema: [{ AttributeName: 'PK', KeyType: 'HASH' }, { AttributeName: 'SK', KeyType: 'RANGE' }],
    });
    template.hasResourceProperties('AWS::DynamoDB::Table', {
      TableName: 'rag-ops-tenants', BillingMode: 'PAY_PER_REQUEST',
      KeySchema: [{ AttributeName: 'key_id', KeyType: 'HASH' }],
    });
  });

  test('Phase 6 declares Secrets Manager and AppConfig contains references only', () => {
    const template = localStack('SecretsStack');
    template.hasResourceProperties('AWS::SecretsManager::Secret', {
      Name: 'rag-ops-guard/runtime/langsmith-api-key',
    });
    const appConfig = JSON.stringify(template.findResources('AWS::AppConfig::HostedConfigurationVersion'));
    expect(appConfig).toContain('langsmith_api_key');
    expect(appConfig).not.toContain('LANGSMITH_API_KEY=');
    expect(appConfig).not.toContain('secret-value');
  });

  test('tenant data role uses trusted principal tag boundaries and Lambda can only assume it', () => {
    const template = localStack('TenantAbacStack', ['tenant-a', 'tenant-b']);
    const roles = JSON.stringify(template.findResources('AWS::IAM::Role'));
    const policies = JSON.stringify(template.findResources('AWS::IAM::Policy'));
    expect(roles).toContain('rag-ops-guard-tenant-data');
    expect(roles).toContain('sts:TagSession');
    expect(policies).toContain('aws:PrincipalTag/tenant_id');
    expect(policies).toContain('dynamodb:LeadingKeys');
    expect(policies).toContain('sts:AssumeRole');
    expect(policies).toContain('sts:TagSession');
    expect(policies).toContain('secretsmanager:GetSecretValue');
    expect(policies).not.toContain('dynamodb:Scan');
  });

  test('Phase 5 admin LeadingKeys and legacy KMS migration resource remain intact', () => {
    const template = localStack('Phase5Compatibility', ['tenant-a', 'tenant-b']);
    template.hasResourceProperties('AWS::KMS::Alias', { AliasName: 'alias/rag-ops-guard-config-secrets' });
    const policies = JSON.stringify(template.findResources('AWS::IAM::Policy'));
    expect(policies).toContain('TENANT#tenant-a');
    expect(policies).toContain('SECRET_SCOPE#TENANT#tenant-a');
    expect(policies).toContain('TENANT#tenant-b');
    expect(policies).toContain('SECRET_SCOPE#TENANT#tenant-b');
  });

  test('AWS target stays fail-closed on external inference endpoints', () => {
    const app = new App();
    expect(() => new RagOpsGuardStack(app, 'InvalidAwsStack', {
      target: 'aws', pythonVersion: '3.13', lambdaCode: inlineCode(),
    })).toThrow(/RAG_OPS_AWS_LLM_BASE_URL/);
  });

  test('AWS target has same environmentless Lambda shape plus dashboard', () => {
    const app = new App();
    const template = Template.fromStack(new RagOpsGuardStack(app, 'AwsTestStack', {
      target: 'aws', pythonVersion: '3.13', lambdaCode: inlineCode(),
      llmBaseUrl: 'https://llm.example.test/v1',
      embeddingBaseUrl: 'https://embedding.example.test/v1',
      rerankerBaseUrl: 'https://reranker.example.test/v1',
    }));
    for (const resource of Object.values(template.findResources('AWS::Lambda::Function')) as Array<{ Properties?: Record<string, unknown> }>) {
      expect(resource.Properties).not.toHaveProperty('Environment');
    }
    template.hasResourceProperties('AWS::CloudWatch::Dashboard', { DashboardName: 'rag-ops-guard-config-admin' });
  });

  test('local target omits CloudWatch dashboard and keeps HTTP routes', () => {
    const template = localStack('LocalFinalStack');
    template.resourceCountIs('AWS::CloudWatch::Dashboard', 0);
    template.hasResourceProperties('AWS::ApiGatewayV2::Route', { RouteKey: 'POST /v1/query' });
    template.hasResourceProperties('AWS::ApiGatewayV2::Route', { RouteKey: 'POST /v1/ingest' });
  });
});
