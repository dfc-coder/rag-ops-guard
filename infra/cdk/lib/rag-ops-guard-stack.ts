import {
  CfnOutput,
  Duration,
  RemovalPolicy,
  Stack,
  StackProps,
  aws_apigatewayv2 as apigwv2,
  aws_dynamodb as dynamodb,
  aws_iam as iam,
  aws_lambda as lambda,
  aws_s3 as s3,
  aws_s3vectors as s3vectors,
} from 'aws-cdk-lib';
import { Construct } from 'constructs';

export type InfraTarget = 'local' | 'aws';

export interface RagOpsGuardStackProps extends StackProps {
  readonly target: InfraTarget;
  readonly pythonVersion: string;
  readonly lambdaCode: lambda.Code;
  readonly configHash?: string;
  readonly llmBaseUrl?: string;
  readonly llmModel?: string;
  readonly embeddingBaseUrl?: string;
  readonly embeddingModel?: string;
  readonly rerankerBaseUrl?: string;
  readonly rerankerModel?: string;
}

function requireHttps(name: string, value: string | undefined): string {
  if (!value || !value.startsWith('https://')) {
    throw new Error(`${name} is required and must use https:// for the AWS target`);
  }
  return value;
}

function targetUrl(
  local: boolean,
  value: string | undefined,
  localDefault: string,
  awsName: string,
): string {
  return local ? (value ?? localDefault) : requireHttps(awsName, value);
}

export class RagOpsGuardStack extends Stack {
  constructor(scope: Construct, id: string, props: RagOpsGuardStackProps) {
    super(scope, id, props);

    const local = props.target === 'local';
    const runtime = new lambda.Runtime(`python${props.pythonVersion}`, lambda.RuntimeFamily.PYTHON, {
      supportsInlineCode: true,
    });

    const documentBucketName = local ? 'rag-ops-guard-docs-local' : undefined;
    const localVectorBucketName = local ? 'rag-ops-guard-vectors-local' : undefined;
    const vectorIndexName = local ? 'ops-knowledge-openvino-v1' : 'ops-knowledge-v1';

    const documents = new s3.Bucket(this, 'Documents', {
      bucketName: documentBucketName,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: !local,
      removalPolicy: RemovalPolicy.RETAIN,
    });

    const vectors = new s3vectors.CfnVectorBucket(
      this,
      'VectorBucket',
      localVectorBucketName ? { vectorBucketName: localVectorBucketName } : {},
    );
    vectors.applyRemovalPolicy(RemovalPolicy.RETAIN);
    const vectorBucketName = localVectorBucketName ?? vectors.ref;

    const index = new s3vectors.CfnIndex(this, 'KnowledgeIndex', {
      vectorBucketName,
      indexName: vectorIndexName,
      dataType: 'float32',
      dimension: 1024,
      distanceMetric: 'cosine',
      metadataConfiguration: { nonFilterableMetadataKeys: ['chunk_s3_key'] },
    });
    index.addDependency(vectors);
    index.applyRemovalPolicy(RemovalPolicy.RETAIN);

    const configTable = new dynamodb.Table(this, 'ConfigTable', {
      tableName: 'rag-ops-config',
      partitionKey: { name: 'PK', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'SK', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      pointInTimeRecovery: true,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
      removalPolicy: RemovalPolicy.RETAIN,
    });

    const tenantTable = new dynamodb.Table(this, 'TenantCredentialTable', {
      tableName: 'rag-ops-tenants',
      partitionKey: { name: 'key_id', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      pointInTimeRecovery: true,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
      removalPolicy: RemovalPolicy.RETAIN,
    });

    const llmBaseUrl = targetUrl(
      local,
      props.llmBaseUrl,
      'http://llama-gen:8080/v1',
      'RAG_OPS_AWS_LLM_BASE_URL',
    );
    const embeddingBaseUrl = targetUrl(
      local,
      props.embeddingBaseUrl,
      'http://rag-ops-ovms-rag:8000/v3',
      'RAG_OPS_AWS_EMBEDDING_BASE_URL',
    );
    const rerankerBaseUrl = targetUrl(
      local,
      props.rerankerBaseUrl,
      'http://rag-ops-ovms-rag:8000/v3',
      'RAG_OPS_AWS_RERANKER_BASE_URL',
    );

    const commonEnvironment = {
      S3_DOCUMENT_BUCKET: documents.bucketName,
      S3_VECTOR_BUCKET: vectorBucketName,
      S3_VECTOR_INDEX: vectorIndexName,
      VECTOR_DIMENSION: '1024',
      CONFIG_SOURCE: 'db',
      CONFIG_TABLE: configTable.tableName,
      TENANT_TABLE: tenantTable.tableName,
      CONFIG_HASH: props.configHash ?? '0'.repeat(64),
      CONFIG_HEAD_TTL_SECONDS: '45',
      CONFIG_MAX_STALE_SECONDS: '300',
      APP_ENV: local ? 'local' : 'aws',
      AWS_ENDPOINT_URL: local ? 'http://floci:4566' : '',
      LLM_BASE_URL: llmBaseUrl,
      LLM_MODEL: props.llmModel ?? 'qwen3.5-2b-unsloth-ud-q4-k-xl',
      EMBEDDING_BASE_URL: embeddingBaseUrl,
      EMBEDDING_MODEL: props.embeddingModel ?? 'OpenVINO/Qwen3-Embedding-0.6B-int8-ov',
      EMBEDDING_DIMENSION: '1024',
      RERANKER_BASE_URL: rerankerBaseUrl,
      RERANKER_MODEL: props.rerankerModel ?? 'OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov',
    };

    const ingest = new lambda.Function(this, 'IngestFunction', {
      functionName: 'rag-ops-guard-ingest',
      runtime,
      handler: 'rag_ops_guard.handlers.ingest.handler',
      code: props.lambdaCode,
      timeout: Duration.seconds(180),
      memorySize: 1024,
      environment: commonEnvironment,
    });
    const query = new lambda.Function(this, 'QueryFunction', {
      functionName: 'rag-ops-guard-query',
      runtime,
      handler: 'rag_ops_guard.handlers.query.handler',
      code: props.lambdaCode,
      timeout: Duration.seconds(180),
      memorySize: 1024,
      environment: commonEnvironment,
    });

    documents.grantReadWrite(ingest);
    documents.grantRead(query);
    tenantTable.grantReadData(ingest);
    tenantTable.grantReadData(query);
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

    const configRead = new iam.PolicyStatement({
      actions: ['dynamodb:GetItem', 'dynamodb:BatchGetItem', 'dynamodb:Query', 'dynamodb:DescribeTable'],
      resources: [configTable.tableArn],
    });
    ingest.addToRolePolicy(configRead);
    query.addToRolePolicy(configRead);

    const api = new apigwv2.CfnApi(this, 'HttpApi', {
      name: local ? 'rag-ops-guard-local' : 'rag-ops-guard',
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
    new CfnOutput(this, 'VectorBucketName', { value: vectorBucketName });
    new CfnOutput(this, 'VectorIndexName', { value: vectorIndexName });
    new CfnOutput(this, 'VectorDimension', { value: '1024' });
    new CfnOutput(this, 'ConfigTableName', { value: configTable.tableName });
    new CfnOutput(this, 'TenantTableName', { value: tenantTable.tableName });
    new CfnOutput(this, 'QueryFunctionName', { value: query.functionName });
    new CfnOutput(this, 'IngestFunctionName', { value: ingest.functionName });
    new CfnOutput(this, 'ApiId', { value: api.ref });
  }
}
