import { useEffect, useState } from "react";
import { Box, Typography, Avatar, Divider, Button } from "@mui/material";
import { Authenticator } from "@aws-amplify/ui-react";
import { fetchUserAttributes } from "aws-amplify/auth";

export default function Account() {
  const [displayName, setDisplayName] = useState<string>("");

  useEffect(() => {
    async function loadName() {
      try {
        const attrs = await fetchUserAttributes();
        setDisplayName(attrs.name || "");
      } catch {
        // user not yet signed in — ignore
      }
    }
    loadName();
  }, []);

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
      {({ signOut, user }) => (
        <Box sx={{ p: 4, maxWidth: 600, margin: "0 auto" }}>
          <Typography variant="h5" fontWeight={700} mb={3}>
            Account
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
            <Avatar sx={{ width: 64, height: 64 }} />
            <Box>
              <Typography fontWeight={600}>
                {displayName || user?.signInDetails?.loginId || user?.username || "Authenticated User"}
              </Typography>
              {displayName && (
                <Typography variant="body2" color="text.secondary">
                  {user?.signInDetails?.loginId}
                </Typography>
              )}
            </Box>
          </Box>
          <Divider sx={{ mb: 3 }} />
          <Typography color="text.secondary" mb={3}>
            Account details coming soon.
          </Typography>
          <Button variant="outlined" color="error" onClick={signOut}>
            Sign Out
          </Button>
        </Box>
      )}
    </Authenticator>
  );
}