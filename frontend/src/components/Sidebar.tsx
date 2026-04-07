import { useState } from "react";
import {
  Button,
  TextField,
  Divider,
  Box,
  Typography,
  Paper,
  Collapse,
} from "@mui/material";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import type { ReactionType } from "./MapComponent";

const REACTIONS: { type: ReactionType; emoji: string; label: string }[] = [
  { type: "icy", emoji: "❄️", label: "Icy" },
  { type: "powder", emoji: "⛷️", label: "Powder" },
  { type: "bluebird", emoji: "☀️", label: "Bluebird" },
  { type: "crowded", emoji: "👥", label: "Crowded" },
  { type: "heavy_snow", emoji: "🌨️", label: "Heavy Snow" },
  { type: "foggy", emoji: "🌫️", label: "Foggy" },
  { type: "sketchy", emoji: "⚠️", label: "Sketchy" },
];

interface SidebarProps {
  onSubmit: (coords: { lat: number; lng: number }) => void;
  sendReaction: (reactionType: ReactionType, message: string) => void;
  pendingLocation: { lat: number; lng: number } | null;
  isLoggedIn: boolean;
}

export default function Sidebar({
  onSubmit,
  sendReaction,
  pendingLocation,
}: SidebarProps) {
  const [lat, setLat] = useState("");
  const [lng, setLng] = useState("");
  const [selectedType, setSelectedType] = useState<ReactionType | null>(null);
  const [message, setMessage] = useState("");
  const [showManualEntry, setShowManualEntry] = useState(false);

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
        },
      );
      if (!response.ok) throw new Error(`API error: ${response.status}`);
      const data = await response.json();
      console.log("API response:", data);
    } catch (err) {
      console.error("Failed to send coordinates:", err);
    }
  };

  const handlePost = () => {
    if (!selectedType || !pendingLocation) return;
    sendReaction(selectedType, message.trim());
    setSelectedType(null);
    setMessage("");
  };

  const canPost = selectedType !== null && pendingLocation !== null;

  const inputSx = {
    "& .MuiOutlinedInput-root": {
      color: "#fff",
      "& fieldset": { borderColor: "rgba(240,248,255,0.15)" },
      "&:hover fieldset": { borderColor: "rgba(240,248,255,0.3)" },
      "&.Mui-focused fieldset": { borderColor: "rgba(240,248,255,0.3)" },
    },
    "& .MuiInputLabel-root": { color: "rgba(240,248,255,0.45)" },
    "& .MuiInputLabel-root.Mui-focused": { color: "rgba(240,248,255,0.3)" },
  };

  return (
    <Paper
      elevation={4}
      sx={{
        width: 350,
        maxWidth: "100%",  // allows it to shrink on mobile inside the sheet
        mx: "auto",
        height: "100%",
        boxSizing: "border-box",
        bgcolor: "#150E2A",
        border: "1px solid rgba(255,45,120,0.15)",
        borderRadius: 3,
        p: 2.5,
        display: "flex",
        flexDirection: "column",
        gap: 2.5,
        overflowY: "auto",
      }}
    >
      {/* ── Drop a Reaction (Now the primary focus) ── */}
      <Box>
        <Typography
          variant="h6"
          mb={0.5}
          sx={{ fontFamily: "'Righteous', sans-serif", color: "#F0F8FF", letterSpacing: "0.03em" }}
        >
          Mountain Conditions
        </Typography>
        <Typography variant="body2" color="rgba(240,248,255,0.55)" mb={2}>
          Click the map to drop a pin, then share your vibe below.
        </Typography>

        <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {/* Pinned location indicator - moved up for logical flow */}
          <Box
            sx={{
              bgcolor: "rgba(240,248,255,0.04)",
              border: "1px solid rgba(240,248,255,0.08)",
              borderRadius: 2,
              p: 1.5,
            }}
          >
            <Typography
              variant="caption"
              color="rgba(240,248,255,0.4)"
              display="block"
              mb={0.5}
              textTransform="uppercase"
              letterSpacing={0.5}
              fontSize={10}
            >
              Selected Location
            </Typography>
            {pendingLocation ? (
              <Typography
                variant="body2"
                color="#fff"
                fontFamily="monospace"
                fontWeight={500}
              >
                {pendingLocation.lat.toFixed(4)},{" "}
                {pendingLocation.lng.toFixed(4)}
              </Typography>
            ) : (
              <Typography
                variant="body2"
                color="rgba(240,248,255,0.4)"
                fontStyle="italic"
              >
                No location selected
              </Typography>
            )}
          </Box>

          {/* Emoji type selector using CSS Grid */}
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 1.25,
            }}
          >
            {REACTIONS.map(({ type, emoji, label }) => (
              <Button
                key={type}
                variant="outlined"
                onClick={() =>
                  setSelectedType((prev) => (prev === type ? null : type))
                }
                sx={{
                  flexDirection: "column",
                  textTransform: "none",
                  fontSize: 12,
                  p: 1.5,
                  lineHeight: 1.2,
                  gap: 0.5,
                  borderRadius: 2,
                  borderColor:
                    selectedType === type
                      ? "rgba(240,248,255,0.8)"
                      : "rgba(240,248,255,0.12)",
                  color:
                    selectedType === type ? "#fff" : "rgba(240,248,255,0.6)",
                  bgcolor:
                    selectedType === type
                      ? "rgba(240,248,255,0.1)"
                      : "transparent",
                  transition: "all 0.2s ease-in-out",
                  "&:hover": {
                    borderColor: "rgba(240,248,255,0.5)",
                    bgcolor: "rgba(240,248,255,0.06)",
                  },
                }}
              >
                <span style={{ fontSize: 20 }}>{emoji}</span>
                {label}
              </Button>
            ))}
          </Box>

          {/* Message input */}
          <TextField
            label="Message (optional)"
            placeholder="What's it like out there?"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            size="small"
            fullWidth
            multiline
            rows={2}
            sx={inputSx}
          />

          {/* Post button */}
          <Button
            variant="contained"
            fullWidth
            disabled={!canPost}
            onClick={handlePost}
            sx={{
              textTransform: "none",
              fontWeight: 600,
              fontSize: 14,
              py: 1.25,
              borderRadius: 2,
              background: "linear-gradient(135deg, #FF2D78 0%, #BF5FFF 100%)",
              boxShadow: "0 0 16px rgba(255,45,120,0.35)",
              "&:hover": {
                background: "linear-gradient(135deg, #FF2D78 0%, #0088FF 100%)",
                boxShadow: "0 0 24px rgba(255,45,120,0.5)",
              },
              "&.Mui-disabled": {
                bgcolor: "rgba(240,248,255,0.05)",
                color: "rgba(240,248,255,0.2)",
                background: "none",
                boxShadow: "none",
              },
            }}
          >
            Post Vibe
          </Button>

          {/* Avalanche reporting disclaimer */}
          <Box
            sx={{
              bgcolor: "rgba(183,28,28,0.15)",
              border: "1px solid rgba(183,28,28,0.4)",
              borderRadius: 2,
              p: 1.5,
            }}
          >
            <Typography
              variant="caption"
              color="rgba(240,248,255,0.9)"
              fontWeight={600}
              display="block"
              mb={0.5}
            >
              💀 Witnessed an avalanche?
            </Typography>
            <Typography
              variant="caption"
              color="rgba(240,248,255,0.65)"
              display="block"
              lineHeight={1.5}
            >
              Please report it directly to the Colorado Avalanche Information
              Center (CAIC). Official reports help keep everyone safe.
            </Typography>
            <Typography
              component="a"
              href="https://avalanche.state.co.us/observations/observation-report"
              target="_blank"
              rel="noopener noreferrer"
              variant="caption"
              sx={{
                color: "#ef9a9a",
                fontWeight: 600,
                textDecoration: "none",
                display: "block",
                mt: 0.75,
                "&:hover": { textDecoration: "underline" },
              }}
            >
              Report on CAIC →
            </Typography>
          </Box>
        </Box>
      </Box>

      <Divider sx={{ borderColor: "rgba(240,248,255,0.08)" }} />

      {/* ── Collapsible Location Search ── */}
      <Box>
        <Button
          fullWidth
          onClick={() => setShowManualEntry(!showManualEntry)}
          endIcon={showManualEntry ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          sx={{
            color: "rgba(240,248,255,0.5)",
            textTransform: "none",
            justifyContent: "space-between",
            px: 1,
            "&:hover": { bgcolor: "rgba(240,248,255,0.05)", color: "#fff" },
          }}
        >
          Enter coordinates manually
        </Button>

        <Collapse in={showManualEntry}>
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              gap: 1.5,
              mt: 2,
              p: 1,
            }}
          >
            <TextField
              label="Latitude"
              placeholder="e.g. 39.7392"
              value={lat}
              onChange={(e) => setLat(e.target.value)}
              size="small"
              fullWidth
              sx={inputSx}
            />
            <TextField
              label="Longitude"
              placeholder="e.g. -104.9903"
              value={lng}
              onChange={(e) => setLng(e.target.value)}
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
                color: "rgba(240,248,255,0.7)",
                borderColor: "rgba(240,248,255,0.2)",
                borderRadius: 2,
                "&:hover": {
                  borderColor: "rgba(240,248,255,0.5)",
                  color: "#fff",
                  bgcolor: "rgba(240,248,255,0.05)",
                },
              }}
            >
              Search Location
            </Button>
          </Box>
        </Collapse>
      </Box>
    </Paper>
  );
}
