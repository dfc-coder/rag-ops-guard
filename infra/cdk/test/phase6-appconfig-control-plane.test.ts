import { App, aws_lambda as lambda } from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import { RagOpsGuardStack } from '../lib/rag-ops-guard-stack';

function inlineCode(): lambda.Code {
  return lambda.Code.fromInline('def handler(event, context):\n    return {"statusCode": 200}\n');
}

describe('Phase 6.0 AppConfig control plane', () => {
  test('canonical CDK creates the hosted control plane and deploys it', () => {
    const app = new App();
    const stack = new RagOpsGuardStack(app, 'Phase6AppConfigStack', {
      target: 'local',
      pythonVersion: '3.13',
      lambdaCode: inlineCode(),
    });
    const template = Template.fromStack(stack);

    template.resourceCountIs('AWS::AppConfig::Application', 1);
    template.hasResourceProperties('AWS::AppConfig::Application', {
      Name: 'rag-ops-guard',
    });
    template.resourceCountIs('AWS::AppConfig::Environment', 1);
    template.hasResourceProperties('AWS::AppConfig::Environment', {
      Name: 'runtime',
    });
    template.resourceCountIs('AWS::AppConfig::ConfigurationProfile', 1);
    template.hasResourceProperties('AWS::AppConfig::ConfigurationProfile', {
      Name: 'control-plane',
      LocationUri: 'hosted',
    });
    template.resourceCountIs('AWS::AppConfig::HostedConfigurationVersion', 1);
    template.hasResourceProperties('AWS::AppConfig::HostedConfigurationVersion', {
      ContentType: 'application/json',
    });
    template.resourceCountIs('AWS::AppConfig::Deployment', 1);
    template.hasResourceProperties('AWS::AppConfig::Deployment', {
      DeploymentStrategyId: 'AppConfig.AllAtOnce',
    });

    const rendered = JSON.stringify(template.toJSON());
    for (const field of [
      'schema_version',
      'config_table',
      'tenant_credential_table',
      'document_bucket',
      'vector_bucket',
      'vector_index_base',
      'llm',
      'embedding',
      'reranker',
      'fail_closed',
    ]) {
      expect(rendered).toContain(field);
    }
    expect(rendered).not.toContain('langsmith_api_key');
    expect(rendered).not.toContain('ragas_judge_api_key');
  });

  test('query and ingest roles can read AppConfigData', () => {
    const app = new App();
    const stack = new RagOpsGuardStack(app, 'Phase6AppConfigIamStack', {
      target: 'local',
      pythonVersion: '3.13',
      lambdaCode: inlineCode(),
    });
    const template = Template.fromStack(stack);
    const policies = JSON.stringify(template.findResources('AWS::IAM::Policy'));

    expect(policies).toContain('appconfig:StartConfigurationSession');
    expect(policies).toContain('appconfig:GetLatestConfiguration');
  });
});
