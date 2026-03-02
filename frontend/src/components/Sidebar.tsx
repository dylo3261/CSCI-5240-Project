import { useState } from "react";
import { Button, TextField, Divider, Box, Typography } from "@mui/material";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import Paper from "@mui/material/Paper";

interface SidebarProps {
  onSubmit: (coords: { lat: number; lng: number }) => void;
}

export default function Sidebar({ onSubmit }: SidebarProps) {
  const [lat, setLat] = useState("");
  const [lng, setLng] = useState("");

  const handleSubmit = async () => {
    onSubmit({ lat: parseFloat(lat), lng: parseFloat(lng) });

    try {
      const response = await fetch(
        "https://mera3wkzuj.execute-api.us-west-2.amazonaws.com/send-coords",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            latitude: parseFloat(lat),
            longitude: parseFloat(lng),
          }),
        }
      );
      if (!response.ok) throw new Error(`API error: ${response.status}`);
      const data = await response.json();
      console.log("API response:", data);
    } catch (err) {
      console.error("Failed to send coordinates:", err);
    }
  };

  return (
    <Paper elevation={4} sx={{
      width: 250,
      flexShrink: 0,
      bgcolor: "#0a1628",
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 3,
      p: 2.5,
      display: "flex",
      flexDirection: "column",
      gap: 2,
      mt: "40px",
    }}>
      <Box>
        <Typography variant="h6" fontWeight={700} color="#fff">
          Location Search
        </Typography>
      </Box>

      <Divider sx={{ borderColor: "rgba(255,255,255,0.08)" }} />

      <TextField
        label="Latitude"
        placeholder="e.g. 39.7392"
        value={lat}
        onChange={e => setLat(e.target.value)}
        size="small"
        fullWidth
        sx={{
          "& .MuiOutlinedInput-root": {
            color: "#fff",
            "& fieldset": { borderColor: "rgba(255,255,255,0.15)" },
            "&:hover fieldset": { borderColor: "rgba(255,255,255,0.3)" },
            "&.Mui-focused fieldset": { borderColor: "rgba(255,255,255,0.3)" },
          },
          "& .MuiInputLabel-root": { color: "rgba(255,255,255,0.45)" },
          "& .MuiInputLabel-root.Mui-focused": { color: "rgba(255,255,255,0.3)" },
        }}
      />

      <TextField
        label="Longitude"
        placeholder="e.g. -104.9903"
        value={lng}
        onChange={e => setLng(e.target.value)}
        size="small"
        fullWidth
        sx={{
          "& .MuiOutlinedInput-root": {
            color: "#fff",
            "& fieldset": { borderColor: "rgba(255,255,255,0.15)" },
            "&:hover fieldset": { borderColor: "rgba(255,255,255,0.3)" },
            "&.Mui-focused fieldset": { borderColor: "rgba(255,255,255,0.3)" },
          },
          "& .MuiInputLabel-root": { color: "rgba(255,255,255,0.45)" },
          "& .MuiInputLabel-root.Mui-focused": { color: "rgba(255,255,255,0.3)" },
        }}
      />

      <Button
        variant="outlined"
        startIcon={<MyLocationIcon />}
        fullWidth
        onClick={handleSubmit}
        sx={{
          textTransform: "none",
          fontWeight: 600,
          color: "rgba(255,255,255,0.6)",
          borderColor: "rgba(255,255,255,0.15)",
          borderRadius: 2,
          "&:hover": {
            borderColor: "rgba(255,255,255,0.4)",
            color: "#fff",
            bgcolor: "rgba(255,255,255,0.05)",
          },
        }}
      >
        Search
      </Button>

      <Divider sx={{ borderColor: "rgba(255,255,255,0.08)" }} />

      <Box sx={{ bgcolor: "rgba(255,255,255,0.04)", borderRadius: 2, p: 1.5 }}>
        <Typography variant="caption" color="rgba(255,255,255,0.3)"
          display="block" mb={0.5}>
          Current Input
        </Typography>
        <Typography variant="caption" color="rgba(255,255,255,0.6)"
          fontFamily="monospace">
          {lat && lng ? `${lat}, ${lng}` : "Nothing entered"}
        </Typography>
      </Box>
    </Paper>
  );
}