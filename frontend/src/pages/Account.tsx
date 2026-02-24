import { useEffect, useState, type Dispatch, type SetStateAction } from "react";
import { Box, Typography, Avatar, Divider, Button, Paper } from "@mui/material";
import { Authenticator } from "@aws-amplify/ui-react";
import { fetchUserAttributes, type FetchUserAttributesOutput, type AuthUser } from "aws-amplify/auth";
import type { UseAuthenticator } from "@aws-amplify/ui-react-core";
import "./Account.css";

// Strict typing for the sub-component props to satisfy linters
interface AccountDetailsProps {
  user?: AuthUser;
  signOut?: UseAuthenticator["signOut"];
  attributes: FetchUserAttributesOutput | null;
  setAttributes: Dispatch<SetStateAction<FetchUserAttributesOutput | null>>;
}

// Custom header injected into the Amplify Authenticator
const components = {
  Header() {
    return (
      <Box className="auth-custom-header">
        <Typography variant="h5" component="h1" className="auth-title">
          Avalanche Predictor
        </Typography>
        <Typography variant="caption" component="p" className="auth-subtitle">
          Backcountry Safety & Intelligence
        </Typography>
      </Box>
    );
  },
};

export default function Account() {
  const [attributes, setAttributes] = useState<FetchUserAttributesOutput | null>(null);

  return (
    <Box className="account-page-wrapper">
      <Authenticator
        loginMechanisms={["email"]}
        components={components}
        signUpAttributes={["name"]}
        formFields={{
          signUp: {
            name: {
              label: "Full Name",
              placeholder: "Enter your full name",
              order: 1,
              isRequired: true,
            },
          },
        }}
      >
        {({ signOut, user }) => (
          <AccountDetails
            user={user}
            signOut={signOut}
            attributes={attributes}
            setAttributes={setAttributes}
          />
        )}
      </Authenticator>
    </Box>
  );
}

// Sub-component handling the authenticated state
function AccountDetails({ user, signOut, attributes, setAttributes }: AccountDetailsProps) {
  useEffect(() => {
    async function loadAttributes() {
      try {
        const attrs = await fetchUserAttributes();
        setAttributes(attrs);
      } catch (err) {
        console.error("Error fetching user attributes:", err);
      }
    }
    
    if (!attributes) {
      loadAttributes();
    }
  }, [attributes, setAttributes]);

  // Safely grab the first letter of the name for the avatar fallback
  const initial = attributes?.name ? attributes.name.charAt(0).toUpperCase() : "U";

  return (
    <Paper elevation={3} className="profile-card">
      <Typography variant="h6" component="h2" className="profile-heading">
        Your Profile
      </Typography>

      <Box className="profile-info-container">
        <Avatar className="profile-avatar">
          {initial}
        </Avatar>
        <Box className="profile-details">
          <Typography variant="subtitle1" component="h3" className="profile-name">
            {attributes?.name || "Loading..."}
          </Typography>
          <Typography variant="body2" className="profile-email">
            {attributes?.email || user?.signInDetails?.loginId}
          </Typography>
        </Box>
      </Box>

      <Divider className="profile-divider" />

      <Box className="profile-meta">
        <Typography variant="caption" component="p" className="meta-label">
          System User ID
        </Typography>
        <Typography variant="body2" component="p" className="meta-value">
          {user?.userId}
        </Typography>
      </Box>

      <Button 
        variant="outlined" 
        fullWidth 
        className="btn-signout" 
        onClick={signOut}
      >
        Secure Sign Out
      </Button>
    </Paper>
  );
}