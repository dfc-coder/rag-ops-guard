import {
  CfnOutput,
  Duration,
  RemovalPolicy,
  Stack,
  StackProps,
  aws_apigatewayv2 as apigwv2,
  aws_appconfig as appconfig,
  aws_cloudwatch as cloudwatch,
  aws_dynamodb as dynamodb,
  aws_iam as iam,
  aws_kms as kms,
  aws_lambda as lambda,
  aws_s3 as s3,
  aws_s3vectors as s3vectors,
  aws_secretsmanager as secretsmanager,
} from 'aws-cdk-lib';
import { Construct } from 'constructs';

export type InfraTarget = 'local' | 'aws';

export interface RagOpsGuardStackProps extends StackProps {
  readonly target: InfraTarget;
  readonly pythonVersion: string;
  readonly lambdaCode: lambda.Code;
  readonly tenantIds?: string[];
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

function tenantIds(values: string[] | undefined): string[] {
  const resolved = [...new Set(values?.length ? values : ['default'])];
  for (const tenantId of resolved) {
    if (!/^[a-z0-9][a-z0-9_-]{0,62}$/.test(tenantId)) {
      throw new Error(`invalid tenant id ${tenantId}`);
    }
  }
  return resolved.sort();
}

export class RagOpsGuardStack extends Stack {
  constructor(scope: Construct, id: string, props: RagOpsGuardStackProps) {
    super(scope, id, props);

    const local = props.target === 'local';
    const runtime = new lambda.Runtime(`python${props.pythonVersion}`, lambda.RuntimeFamily.PYTHON, {
      supportsInlineCode: true,
    });
    const tenants = tenantIds(props.tenantIds);
    const documentBucketName = local ? 'rag-ops-guard-docs-local' : undefined;
    const localVectorBucketName = local ? 'rag-ops-guard-vectors-local' : undefined;
    const vectorIndexBaseName = local ? 'ops-knowledge-openvino-v1' : 'ops-knowledge-v1';
    const tenantVectorIndexes = tenants.map((tenantId) => `${vectorIndexBaseName}--${tenantId}`);

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
    tenantVectorIndexes.forEach((indexName, position) => {
      const index = new s3vectors.CfnIndex(this, `KnowledgeIndex${position}`, {
        vectorBucketName,
        indexName,
        dataType: 'float32',
        dimension: 1024,
        distanceMetric: 'cosine',
        metadataConfiguration: { nonFilterableMetadataKeys: ['chunk_s3_key'] },
      });
      index.addDependency(vectors);
      index.applyRemovalPolicy(RemovalPolicy.RETAIN);
    });

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

    const configSecretsKey = new kms.Key(this, 'ConfigSecretsKey', {
      description: 'Legacy Phase-3/5 envelope KEK retained only for migration history',
      enableKeyRotation: !local,
      removalPolicy: RemovalPolicy.RETAIN,
    });
    const configSecretsAlias = new kms.Alias(this, 'ConfigSecretsKeyAlias', {
      aliasName: 'alias/rag-ops-guard-config-secrets',
      targetKey: configSecretsKey,
    });

    tenants.forEach((tenantId, position) => {
      const tenantConfigAdmin = new iam.Role(this, `TenantConfigAdmin${position}`, {
        assumedBy: new iam.AccountPrincipal(this.account),
        description: `Administrative config role restricted to tenant ${tenantId}`,
      });
      tenantConfigAdmin.addToPolicy(
        new iam.PolicyStatement({
          actions: [
            'dynamodb:GetItem', 'dynamodb:BatchGetItem', 'dynamodb:Query', 'dynamodb:PutItem',
            'dynamodb:UpdateItem', 'dynamodb:DeleteItem', 'dynamodb:TransactWriteItems',
          ],
          resources: [configTable.tableArn],
          conditions: {
            'ForAllValues:StringEquals': {
              'dynamodb:LeadingKeys': [`TENANT#${tenantId}`, `SECRET_SCOPE#TENANT#${tenantId}`],
            },
          },
        }),
      );
      tenantConfigAdmin.addToPolicy(new iam.PolicyStatement({
        actions: ['dynamodb:DescribeTable'], resources: [configTable.tableArn],
      }));
      configSecretsKey.grantEncryptDecrypt(tenantConfigAdmin);
    });

    const tenantDataRole = new iam.Role(this, 'TenantDataRole', {
      roleName: 'rag-ops-guard-tenant-data',
      assumedBy: new iam.AccountPrincipal(this.account),
      description: 'Tenant data plane role assumed only after API-key authentication',
      maxSessionDuration: Duration.hours(1),
    });
    tenantDataRole.assumeRolePolicy?.addStatements(new iam.PolicyStatement({
      actions: ['sts:TagSession'],
      principals: [new iam.AccountPrincipal(this.account)],
    }));
    tenantDataRole.addToPolicy(new iam.PolicyStatement({
      actions: ['dynamodb:GetItem', 'dynamodb:BatchGetItem', 'dynamodb:Query'],
      resources: [configTable.tableArn],
      conditions: {
        'ForAllValues:StringEquals': {
          'dynamodb:LeadingKeys': ['TENANT#${aws:PrincipalTag/tenant_id}'],
        },
      },
    }));
    tenantDataRole.addToPolicy(new iam.PolicyStatement({
      actions: ['dynamodb:DescribeTable'], resources: [configTable.tableArn],
    }));
    tenantDataRole.addToPolicy(new iam.PolicyStatement({
      actions: ['s3:GetObject', 's3:PutObject', 's3:DeleteObject'],
      resources: [`${documents.bucketArn}/t/\${aws:PrincipalTag/tenant_id}/*`],
    }));
    tenantDataRole.addToPolicy(new iam.PolicyStatement({
      actions: ['s3:ListBucket'],
      resources: [documents.bucketArn],
      conditions: { StringLike: { 's3:prefix': ['t/${aws:PrincipalTag/tenant_id}/*'] } },
    }));
    tenantDataRole.addToPolicy(new iam.PolicyStatement({
      actions: ['s3vectors:PutVectors', 's3vectors:GetVectors', 's3vectors:DeleteVectors', 's3vectors:QueryVectors'],
      resources: ['*'],
    }));
    tenantDataRole.addToPolicy(new iam.PolicyStatement({
      actions: ['secretsmanager:GetSecretValue'],
      resources: [`arn:${this.partition}:secretsmanager:${this.region}:${this.account}:secret:rag-ops-guard/tenant/\${aws:PrincipalTag/tenant_id}/*`],
    }));

    const runtimeLangSmithSecret = new secretsmanager.Secret(this, 'RuntimeLangSmithSecret', {
      secretName: 'rag-ops-guard/runtime/langsmith-api-key',
      description: 'Operator-rotated LangSmith API key; tracing is disabled until explicitly enabled',
      generateSecretString: { excludePunctuation: true, passwordLength: 32 },
    });

    const llmBaseUrl = targetUrl(local, props.llmBaseUrl, 'http://llama-gen:8080/v1', 'RAG_OPS_AWS_LLM_BASE_URL');
    const embeddingBaseUrl = targetUrl(local, props.embeddingBaseUrl, 'http://rag-ops-ovms-rag:8000/v3', 'RAG_OPS_AWS_EMBEDDING_BASE_URL');
    const rerankerBaseUrl = targetUrl(local, props.rerankerBaseUrl, 'http://rag-ops-ovms-rag:8000/v3', 'RAG_OPS_AWS_RERANKER_BASE_URL');
    const llmModel = props.llmModel ?? 'qwen3.5-2b-unsloth-ud-q4-k-xl';
    const embeddingModel = props.embeddingModel ?? 'OpenVINO/Qwen3-Embedding-0.6B-int8-ov';
    const rerankerModel = props.rerankerModel ?? 'OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov';

    const controlPlaneApplication = new appconfig.CfnApplication(this, 'ControlPlaneApplication', {
      name: 'rag-ops-guard', description: 'RAG Ops Guard platform discovery control plane',
    });
    const controlPlaneEnvironment = new appconfig.CfnEnvironment(this, 'ControlPlaneEnvironment', {
      applicationId: controlPlaneApplication.ref, name: 'runtime', description: 'Runtime workload control plane',
    });
    const controlPlaneProfile = new appconfig.CfnConfigurationProfile(this, 'ControlPlaneProfile', {
      applicationId: controlPlaneApplication.ref, name: 'control-plane', locationUri: 'hosted',
      description: 'Non-secret platform discovery configuration',
    });
    const controlPlanePayload = this.toJsonString({
      schema_version: 1,
      resources: {
        config_table: configTable.tableName,
        tenant_credential_table: tenantTable.tableName,
        document_bucket: documents.bucketName,
        vector_bucket: vectorBucketName,
        vector_index_base: vectorIndexBaseName,
        tenant_data_role_arn: tenantDataRole.roleArn,
      },
      services: {
        llm: { base_url: llmBaseUrl, model: llmModel },
        embedding: { base_url: embeddingBaseUrl, model: embeddingModel, dimension: 1024 },
        reranker: { base_url: rerankerBaseUrl, model: rerankerModel },
      },
      secret_refs: { langsmith_api_key: runtimeLangSmithSecret.secretArn },
      bootstrap: { config_head_ttl_seconds: 45, config_max_stale_seconds: 300, fail_closed: true },
    });
    const controlPlaneVersion = new appconfig.CfnHostedConfigurationVersion(this, 'ControlPlaneHostedVersion', {
      applicationId: controlPlaneApplication.ref,
      configurationProfileId: controlPlaneProfile.ref,
      content: controlPlanePayload,
      contentType: 'application/json',
      description: 'Schema v1 RAG Ops Guard platform discovery payload',
      versionLabel: 'schema-v1',
    });
    const controlPlaneDeployment = new appconfig.CfnDeployment(this, 'ControlPlaneDeployment', {
      applicationId: controlPlaneApplication.ref,
      environmentId: controlPlaneEnvironment.ref,
      configurationProfileId: controlPlaneProfile.ref,
      configurationVersion: controlPlaneVersion.ref,
      deploymentStrategyId: 'AppConfig.AllAtOnce',
      description: 'Deploy the active RAG Ops Guard runtime control plane',
    });
    controlPlaneDeployment.addResourceDependency(controlPlaneVersion);

    const ingest = new lambda.Function(this, 'IngestFunction', {
      functionName: 'rag-ops-guard-ingest', runtime,
      handler: 'rag_ops_guard.handlers.ingest.handler', code: props.lambdaCode,
      timeout: Duration.seconds(180), memorySize: 1024,
    });
    const query = new lambda.Function(this, 'QueryFunction', {
      functionName: 'rag-ops-guard-query', runtime,
      handler: 'rag_ops_guard.handlers.query.handler', code: props.lambdaCode,
      timeout: Duration.seconds(180), memorySize: 1024,
    });

    const controlPlaneDiscover = new iam.PolicyStatement({
      actions: ['appconfig:ListApplications', 'appconfig:ListEnvironments', 'appconfig:ListConfigurationProfiles'],
      resources: ['*'],
    });
    const controlPlaneConfigurationArn = `arn:${this.partition}:appconfig:${this.region}:${this.account}:application/${controlPlaneApplication.ref}/environment/${controlPlaneEnvironment.ref}/configuration/${controlPlaneProfile.ref}`;
    const controlPlaneRead = new iam.PolicyStatement({
      actions: ['appconfig:StartConfigurationSession', 'appconfig:GetLatestConfiguration'],
      resources: [controlPlaneConfigurationArn],
    });
    const tenantCredentialRead = new iam.PolicyStatement({
      actions: ['dynamodb:GetItem', 'dynamodb:DescribeTable'], resources: [tenantTable.tableArn],
    });
    const assumeTenantData = new iam.PolicyStatement({
      actions: ['sts:AssumeRole', 'sts:TagSession'], resources: [tenantDataRole.roleArn],
    });
    for (const fn of [ingest, query]) {
      fn.addToRolePolicy(controlPlaneDiscover);
      fn.addToRolePolicy(controlPlaneRead);
      fn.addToRolePolicy(tenantCredentialRead);
      fn.addToRolePolicy(assumeTenantData);
      runtimeLangSmithSecret.grantRead(fn);
    }

    if (!local) {
      const metric = (metricName: string, statistic = 'Sum') => new cloudwatch.Metric({
        namespace: 'RagOpsGuard', metricName, statistic, period: Duration.minutes(5),
      });
      const dashboard = new cloudwatch.Dashboard(this, 'ConfigAdminDashboard', {
        dashboardName: 'rag-ops-guard-config-admin',
      });
      dashboard.addWidgets(
        new cloudwatch.GraphWidget({ title: 'Configuration resolution', left: [metric('ConfigResolveLatencyMs', 'p99'), metric('ConfigRevisionAge', 'Maximum')] }),
        new cloudwatch.GraphWidget({ title: 'Configuration cache and availability', left: [metric('ConfigCacheHit'), metric('ConfigCacheMiss'), metric('ConfigStaleServed'), metric('ConfigDbUnavailable')] }),
        new cloudwatch.GraphWidget({ title: 'Security signals', left: [metric('SecretDecryptFailure'), metric('TenantIsolationViolation')] }),
      );
    }

    const api = new apigwv2.CfnApi(this, 'HttpApi', {
      name: local ? 'rag-ops-guard-local' : 'rag-ops-guard', protocolType: 'HTTP',
    });
    const queryIntegration = new apigwv2.CfnIntegration(this, 'QueryIntegration', {
      apiId: api.ref, integrationType: 'AWS_PROXY', integrationUri: query.functionArn, payloadFormatVersion: '2.0',
    });
    const ingestIntegration = new apigwv2.CfnIntegration(this, 'IngestIntegration', {
      apiId: api.ref, integrationType: 'AWS_PROXY', integrationUri: ingest.functionArn, payloadFormatVersion: '2.0',
    });
    new apigwv2.CfnRoute(this, 'QueryRoute', { apiId: api.ref, routeKey: 'POST /v1/query', target: `integrations/${queryIntegration.ref}` });
    new apigwv2.CfnRoute(this, 'IngestRoute', { apiId: api.ref, routeKey: 'POST /v1/ingest', target: `integrations/${ingestIntegration.ref}` });
    new apigwv2.CfnStage(this, 'DefaultStage', { apiId: api.ref, stageName: '$default', autoDeploy: true });
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
    new CfnOutput(this, 'VectorIndexName', { value: vectorIndexBaseName });
    new CfnOutput(this, 'TenantVectorIndexNames', { value: JSON.stringify(tenantVectorIndexes) });
    new CfnOutput(this, 'VectorDimension', { value: '1024' });
    new CfnOutput(this, 'ConfigTableName', { value: configTable.tableName });
    new CfnOutput(this, 'ConfigSecretsKeyAliasOutput', { value: configSecretsAlias.aliasName });
    new CfnOutput(this, 'TenantTableName', { value: tenantTable.tableName });
    new CfnOutput(this, 'AppConfigApplicationName', { value: 'rag-ops-guard' });
    new CfnOutput(this, 'AppConfigEnvironmentName', { value: 'runtime' });
    new CfnOutput(this, 'AppConfigProfileName', { value: 'control-plane' });
    new CfnOutput(this, 'AppConfigControlPlanePayload', { value: controlPlanePayload });
    new CfnOutput(this, 'QueryFunctionName', { value: query.functionName });
    new CfnOutput(this, 'IngestFunctionName', { value: ingest.functionName });
    new CfnOutput(this, 'ApiId', { value: api.ref });
  }
}
