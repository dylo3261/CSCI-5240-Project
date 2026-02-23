import { ResourcesConfig } from 'aws-amplify';
//fill these in
export const awsConfig: ResourcesConfig = {
  Auth: {
    Cognito: {
      userPoolId: 'us-east-2_000000000',
      userPoolClientId: 'abcdefg', 
    }
  }
};