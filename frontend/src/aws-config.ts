import { ResourcesConfig } from 'aws-amplify';
//fill these in
export const awsConfig: ResourcesConfig = {
  Auth: {
    Cognito: {
      userPoolId: 'us-west-2_kDpA7w9oG',
      userPoolClientId: '2k7n7bsjnpp1ei6m0ul7661ccb',
    }
  }
};