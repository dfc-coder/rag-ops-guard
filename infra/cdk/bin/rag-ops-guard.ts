#!/usr/bin/env node
import { App } from 'aws-cdk-lib';
import { RagOpsGuardStack } from '../lib/rag-ops-guard-stack';

const app = new App();
new RagOpsGuardStack(app, 'RagOpsGuardStack');
