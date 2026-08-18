import {
  CfnOutput,
  Duration,
  RemovalPolicy,
  Stack,
  StackProps,
  aws_apigatewayv2 as apigwv2,
  aws_iam as iam,
  aws_lambda as lambda,
  aws_s3 as s3,
  aws_s3vectors as s3vectors,
} from 'aws-cdk-lib';
import { Construct } from 'constructs';
import path from 'node:path';

export class RagOpsGuardStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const documents = new s3.Bucket(this, 'Documents', {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      removalPolicy: RemovalPolicy.RETAIN,
    });

    const vectors = new s3vectors.CfnVectorBucket(this, 'VectorBucket');
    const index = new s3vectors.CfnIndex(this, 'KnowledgeIndex', {
      vectorBucketName: vectors.ref,
      indexName: 'ops-knowledge-v1',
      dataType: 'float32',
      dimension: 1024,
      distanceMetric: 'cosine',
      metadataConfiguration: {
        nonFilterableMetadataKeys: ['chunk_s3_key'],
      },
    });
    index.addDependency(vectors);

    const code = lambda.Code.fromAsset(path.join(__dirname, '..', 'lambda-placeholder'));
    const commonEnvironment = {
      S3_DOCUMENT_BUCKET: documents.bucketName,
      S3_VECTOR_BUCKET: vectors.ref,
      S3_VECTOR_INDEX: 'ops-knowledge-v1',
      VECTOR_DIMENSION: '1024',
      APP_ENV: 'aws',
      AWS_ENDPOINT_URL: '',
    };

    const ingest = new lambda.Function(this, 'IngestFunction', {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      code,
      timeout: Duration.seconds(120),
      memorySize: 1024,
      environment: commonEnvironment,
    });
    const query = new lambda.Function(this, 'QueryFunction', {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      code,
      timeout: Duration.seconds(120),
      memorySize: 1024,
      environment: commonEnvironment,
    });

    documents.grantReadWrite(ingest);
    documents.grantRead(query);
    ingest.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ['s3vectors:PutVectors', 's3vectors:GetVectors', 's3vectors:DeleteVectors'],
        resources: ['*'],
      }),
    );
    query.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ['s3vectors:QueryVectors', 's3vectors:GetVectors'],
        resources: ['*'],
      }),
    );

    const api = new apigwv2.CfnApi(this, 'HttpApi', {
      name: 'rag-ops-guard',
      protocolType: 'HTTP',
    });

    const queryIntegration = new apigwv2.CfnIntegration(this, 'QueryIntegration', {
      apiId: api.ref,
      integrationType: 'AWS_PROXY',
      integrationUri: query.functionArn,
      payloadFormatVersion: '2.0',
    });
    const ingestIntegration = new apigwv2.CfnIntegration(this, 'IngestIntegration', {
      apiId: api.ref,
      integrationType: 'AWS_PROXY',
      integrationUri: ingest.functionArn,
      payloadFormatVersion: '2.0',
    });

    new apigwv2.CfnRoute(this, 'QueryRoute', {
      apiId: api.ref,
      routeKey: 'POST /v1/query',
      target: `integrations/${queryIntegration.ref}`,
    });
    new apigwv2.CfnRoute(this, 'IngestRoute', {
      apiId: api.ref,
      routeKey: 'POST /v1/ingest',
      target: `integrations/${ingestIntegration.ref}`,
    });
    new apigwv2.CfnStage(this, 'DefaultStage', {
      apiId: api.ref,
      stageName: '$default',
      autoDeploy: true,
    });

    query.addPermission('ApiGatewayQueryInvoke', {
      principal: new iam.ServicePrincipal('apigateway.amazonaws.com'),
      sourceArn: `arn:${this.partition}:execute-api:${this.region}:${this.account}:${api.ref}/*/*`,
    });
    ingest.addPermission('ApiGatewayIngestInvoke', {
      principal: new iam.ServicePrincipal('apigateway.amazonaws.com'),
      sourceArn: `arn:${this.partition}:execute-api:${this.region}:${this.account}:${api.ref}/*/*`,
    });

    new CfnOutput(this, 'DocumentBucketName', { value: documents.bucketName });
    new CfnOutput(this, 'VectorBucketName', { value: vectors.ref });
    new CfnOutput(this, 'VectorIndexName', { value: 'ops-knowledge-v1' });
  }
}
