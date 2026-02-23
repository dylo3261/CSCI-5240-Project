import { Box, Typography, Avatar, Divider, Button } from "@mui/material";
import { Authenticator } from "@aws-amplify/ui-react";

export default function Account() {
  return (
    <Authenticator>
      {({ signOut, user }) => (
        <Box sx={{ p: 4, maxWidth: 600, margin: "0 auto" }}>
          <Typography variant="h5" fontWeight={700} mb={3}>
            Account
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
            <Avatar sx={{ width: 64, height: 64 }} />
            <Box>
              <Typography fontWeight={600}>
                {user?.signInDetails?.loginId || user?.username || "Authenticated User"}
              </Typography>
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