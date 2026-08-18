import { App } from 'aws-cdk-lib';
import { Match, Template } from 'aws-cdk-lib/assertions';
import { RagOpsGuardStack } from '../lib/rag-ops-guard-stack';

describe('RagOpsGuardStack', () => {
  const app = new App();
  const stack = new RagOpsGuardStack(app, 'TestStack');
  const template = Template.fromStack(stack);

  test('creates one S3 Vectors bucket and 1024-dimensional cosine index', () => {
    template.resourceCountIs('AWS::S3Vectors::VectorBucket', 1);
    template.hasResourceProperties('AWS::S3Vectors::Index', {
      DataType: 'float32',
      Dimension: 1024,
      DistanceMetric: 'cosine',
      IndexName: 'ops-knowledge-v1',
    });
  });

  test('creates query and ingest lambdas', () => {
    template.resourceCountIs('AWS::Lambda::Function', 2);
  });

  test('creates retained pay-per-request config table', () => {
    template.resourceCountIs('AWS::DynamoDB::Table', 1);
    template.hasResourceProperties('AWS::DynamoDB::Table', {
      TableName: 'rag-ops-config',
      BillingMode: 'PAY_PER_REQUEST',
      PointInTimeRecoverySpecification: { PointInTimeRecoveryEnabled: true },
      KeySchema: [
        { AttributeName: 'PK', KeyType: 'HASH' },
        { AttributeName: 'SK', KeyType: 'RANGE' },
      ],
    });
  });

  test('declares real AWS endpoint and config table semantics for both lambdas', () => {
    template.resourcePropertiesCountIs(
      'AWS::Lambda::Function',
      {
        Environment: {
          Variables: Match.objectLike({
            APP_ENV: 'aws',
            AWS_ENDPOINT_URL: '',
            CONFIG_TABLE: Match.anyValue(),
          }),
        },
      },
      2,
    );
  });

  test('config table policy never grants Scan', () => {
    const policies = template.findResources('AWS::IAM::Policy');
    const serialized = JSON.stringify(policies);
    expect(serialized).toContain('dynamodb:GetItem');
    expect(serialized).toContain('dynamodb:Query');
    expect(serialized).not.toContain('dynamodb:Scan');
  });

  test('creates HTTP API routes', () => {
    template.hasResourceProperties('AWS::ApiGatewayV2::Route', {
      RouteKey: 'POST /v1/query',
    });
    template.hasResourceProperties('AWS::ApiGatewayV2::Route', {
      RouteKey: 'POST /v1/ingest',
    });
  });
});
