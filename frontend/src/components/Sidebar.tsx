import { useState } from "react";
import { Button, TextField, Divider, Box, Typography } from "@mui/material";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import Paper from "@mui/material/Paper";
import type { ReactionType } from "./MapComponent";

const REACTIONS: { type: ReactionType; emoji: string; label: string }[] = [
  { type: "icy",              emoji: "❄️", label: "Icy"      },
  { type: "good_snow",        emoji: "✨", label: "Good Snow" },
  { type: "avalanche_danger", emoji: "⚠️", label: "Danger"   },
];

interface SidebarProps {
  onSubmit: (coords: { lat: number; lng: number }) => void;
  sendReaction: (reactionType: ReactionType, message: string) => void;
  pendingLocation: { lat: number; lng: number } | null;
  isLoggedIn: boolean;
}

export default function Sidebar({ onSubmit, sendReaction, pendingLocation, isLoggedIn }: SidebarProps) {
  const [lat, setLat] = useState("");
  const [lng, setLng] = useState("");
  const [selectedType, setSelectedType] = useState<ReactionType | null>(null);
  const [message, setMessage] = useState("");

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

  const handlePost = () => {
    if (!selectedType || !pendingLocation || !message.trim()) return;
    sendReaction(selectedType, message.trim());
    setSelectedType(null);
    setMessage("");
  };

  const canPost = selectedType !== null && pendingLocation !== null && message.trim().length > 0;

  const inputSx = {
    "& .MuiOutlinedInput-root": {
      color: "#fff",
      "& fieldset": { borderColor: "rgba(255,255,255,0.15)" },
      "&:hover fieldset": { borderColor: "rgba(255,255,255,0.3)" },
      "&.Mui-focused fieldset": { borderColor: "rgba(255,255,255,0.3)" },
    },
    "& .MuiInputLabel-root": { color: "rgba(255,255,255,0.45)" },
    "& .MuiInputLabel-root.Mui-focused": { color: "rgba(255,255,255,0.3)" },
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
      overflowY: "auto",
    }}>

      {/* ── Location Search ── */}
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
        sx={inputSx}
      />

      <TextField
        label="Longitude"
        placeholder="e.g. -104.9903"
        value={lng}
        onChange={e => setLng(e.target.value)}
        size="small"
        fullWidth
        sx={inputSx}
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
        <Typography variant="caption" color="rgba(255,255,255,0.6)" fontFamily="monospace">
          {lat && lng ? `${lat}, ${lng}` : "Nothing entered"}
        </Typography>
      </Box>

      <Divider sx={{ borderColor: "rgba(255,255,255,0.08)" }} />

      {/* ── Drop a Reaction ── */}
      <Box>
        <Typography variant="caption" fontWeight={600} color="rgba(255,255,255,0.45)"
          display="block" mb={1.5} textTransform="uppercase" letterSpacing={0.5} fontSize={10}>
          Drop a Reaction
        </Typography>

        {!isLoggedIn ? (
          <Box sx={{
            bgcolor: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.06)",
            borderRadius: 2,
            p: 1.5,
            textAlign: "center",
          }}>
            <Typography variant="caption" color="rgba(255,255,255,0.25)" fontSize={11}>
              Sign in to post reactions
            </Typography>
          </Box>
        ) : (
          <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>

            {/* Emoji type selector */}
            <Box sx={{ display: "flex", gap: 1 }}>
              {REACTIONS.map(({ type, emoji, label }) => (
                <Button
                  key={type}
                  variant="outlined"
                  onClick={() => setSelectedType(prev => prev === type ? null : type)}
                  sx={{
                    flex: 1,
                    flexDirection: "column",
                    textTransform: "none",
                    fontSize: 10,
                    px: 0.5,
                    py: 0.75,
                    minWidth: 0,
                    lineHeight: 1.4,
                    gap: 0.25,
                    borderRadius: 2,
                    borderColor: selectedType === type
                      ? "rgba(255,255,255,0.6)"
                      : "rgba(255,255,255,0.12)",
                    color: selectedType === type ? "#fff" : "rgba(255,255,255,0.5)",
                    bgcolor: selectedType === type ? "rgba(255,255,255,0.08)" : "transparent",
                    "&:hover": {
                      borderColor: "rgba(255,255,255,0.4)",
                      bgcolor: "rgba(255,255,255,0.06)",
                    },
                  }}
                >
                  <span style={{ fontSize: 16 }}>{emoji}</span>
                  {label}
                </Button>
              ))}
            </Box>

            {/* Message input */}
            <TextField
              label="Message"
              placeholder="What's the condition?"
              value={message}
              onChange={e => setMessage(e.target.value)}
              size="small"
              fullWidth
              multiline
              maxRows={3}
              sx={inputSx}
            />

            {/* Pinned location indicator */}
            <Box sx={{ bgcolor: "rgba(255,255,255,0.04)", borderRadius: 2, p: 1.25 }}>
              <Typography variant="caption" color="rgba(255,255,255,0.3)"
                display="block" mb={0.25}>
                Pinned Location
              </Typography>
              {pendingLocation ? (
                <Typography variant="caption" color="rgba(255,255,255,0.7)"
                  fontFamily="monospace" fontSize={11}>
                  {pendingLocation.lat.toFixed(4)}, {pendingLocation.lng.toFixed(4)}
                </Typography>
              ) : (
                <Typography variant="caption" color="rgba(255,255,255,0.25)" fontSize={11}>
                  Click the map to place a pin
                </Typography>
              )}
            </Box>

            {/* Post button */}
            <Button
              variant="contained"
              fullWidth
              disabled={!canPost}
              onClick={handlePost}
              sx={{
                textTransform: "none",
                fontWeight: 600,
                fontSize: 13,
                borderRadius: 2,
                bgcolor: "#1565c0",
                "&:hover": { bgcolor: "#1976d2" },
                "&.Mui-disabled": {
                  bgcolor: "rgba(255,255,255,0.05)",
                  color: "rgba(255,255,255,0.2)",
                },
              }}
            >
              Post Reaction
            </Button>

          </Box>
        )}
      </Box>

    </Paper>
  );
}
