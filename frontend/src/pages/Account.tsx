import { Box, Typography, Avatar, Divider } from "@mui/material";

export default function Account() {
  return (
    <Box sx={{ p: 4, maxWidth: 600, margin: "0 auto" }}>
      <Typography variant="h5" fontWeight={700} mb={3}>
        Account
      </Typography>
      <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
        <Avatar sx={{ width: 64, height: 64 }} />
        <Box>
          <Typography fontWeight={600}>Your Name</Typography>
          <Typography variant="body2" color="text.secondary">your@email.com</Typography>
        </Box>
      </Box>
      <Divider sx={{ mb: 3 }} />
      <Typography color="text.secondary">
        Account details coming soon.
      </Typography>
    </Box>
  );
}