import { useEffect, useState } from "react";
import { Box, Typography, Avatar, Divider, Button } from "@mui/material";
import { Authenticator } from "@aws-amplify/ui-react";
import { fetchUserAttributes, type FetchUserAttributesOutput } from "aws-amplify/auth";

export default function Account() {
  // Store the full attributes object to access email, name, etc.
  const [attributes, setAttributes] = useState<FetchUserAttributesOutput | null>(null);

  return (
    <Authenticator
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
      {({ signOut, user }) => {
        // This inner component handles fetching once authenticated
        return (
          <AccountDetails 
            user={user} 
            signOut={signOut} 
            attributes={attributes} 
            setAttributes={setAttributes} 
          />
        );
      }}
    </Authenticator>
  );
}

// Sub-component to handle the "Logged In" state logic
function AccountDetails({ user, signOut, attributes, setAttributes }: any) {
  useEffect(() => {
    async function loadAttributes() {
      try {
        const attrs = await fetchUserAttributes();
        setAttributes(attrs);
      } catch (err) {
        console.error("Error fetching attributes", err);
      }
    }
    // Only fetch if we don't have them yet
    if (!attributes) {
      loadAttributes();
    }
  }, [attributes, setAttributes]);

  return (
    <Box sx={{ p: 4, maxWidth: 600, margin: "0 auto" }}>
      <Typography variant="h5" fontWeight={700} mb={3}>
        Account
      </Typography>
      
      <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
        {/* You can use the first letter of their name as a fallback for the Avatar */}
        <Avatar sx={{ width: 64, height: 64, bgcolor: 'primary.main' }}>
          {attributes?.name?.charAt(0) || "U"}
        </Avatar>
        <Box>
          <Typography fontWeight={600} variant="h6">
            {attributes?.name || "Loading name..."}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {attributes?.email || user?.signInDetails?.loginId}
          </Typography>
        </Box>
      </Box>

      <Divider sx={{ mb: 3 }} />

      <Box sx={{ mb: 4 }}>
        <Typography variant="subtitle2" color="text.secondary">User ID</Typography>
        <Typography variant="body1" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
          {user?.userId}
        </Typography>
      </Box>

      <Button variant="contained" color="error" onClick={signOut}>
        Sign Out
      </Button>
    </Box>
  );
}