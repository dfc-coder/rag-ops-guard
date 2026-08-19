#!/usr/bin/env node
import { App, aws_lambda as lambda } from 'aws-cdk-lib';
import fs from 'node:fs';
import path from 'node:path';
import { InfraTarget, RagOpsGuardStack } from '../lib/rag-ops-guard-stack';

const app = new App();
const root = path.resolve(__dirname, '../../..');
const pythonVersion = fs.readFileSync(path.join(root, '.python-version'), 'utf8').trim();
const targetRaw = process.env.RAG_OPS_INFRA_TARGET ?? 'local';
if (targetRaw !== 'local' && targetRaw !== 'aws') {
  throw new Error(`RAG_OPS_INFRA_TARGET must be local or aws; found ${targetRaw}`);
}
const target: InfraTarget = targetRaw;

const lambdaAssetPath = process.env.RAG_OPS_LAMBDA_ASSET ?? path.join(root, '.local/lambda-package.zip');
if (!fs.existsSync(lambdaAssetPath)) {
  throw new Error(`Lambda asset not found at ${lambdaAssetPath}; run make package-lambda first`);
}

const local = target === 'local';
const configuredTenants = (process.env.RAG_OPS_TENANTS ?? process.env.RAG_OPS_TENANT_ID ?? 'default')
  .split(',')
  .map((value) => value.trim())
  .filter(Boolean);
const stackId = local ? 'RagOpsGuardLocal' : 'RagOpsGuardAws';
new RagOpsGuardStack(app, stackId, {
  target,
  pythonVersion,
  lambdaCode: lambda.Code.fromAsset(lambdaAssetPath),
  tenantIds: configuredTenants,
  configHash: process.env.CONFIG_HASH,
  llmBaseUrl: local ? process.env.LAMBDA_LLM_BASE_URL : process.env.RAG_OPS_AWS_LLM_BASE_URL,
  llmModel: process.env.LLM_MODEL,
  embeddingBaseUrl: local
    ? process.env.LAMBDA_EMBEDDING_BASE_URL
    : process.env.RAG_OPS_AWS_EMBEDDING_BASE_URL,
  embeddingModel: local ? process.env.OVMS_EMBEDDING_MODEL : process.env.EMBEDDING_MODEL,
  rerankerBaseUrl: local
    ? process.env.LAMBDA_RERANKER_BASE_URL
    : process.env.RAG_OPS_AWS_RERANKER_BASE_URL,
  rerankerModel: local ? process.env.OVMS_RERANKER_MODEL : process.env.RERANKER_MODEL,
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION ?? process.env.AWS_REGION ?? 'us-east-1',
  },
});
