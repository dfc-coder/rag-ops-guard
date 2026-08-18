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

  test('declares real AWS endpoint semantics for both lambdas', () => {
    template.resourcePropertiesCountIs(
      'AWS::Lambda::Function',
      {
        Environment: {
          Variables: Match.objectLike({
            APP_ENV: 'aws',
            AWS_ENDPOINT_URL: '',
          }),
        },
      },
      2,
    );
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
