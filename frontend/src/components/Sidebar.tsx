import { useState } from "react";
import { 
  Button, 
  TextField, 
  Divider, 
  Box, 
  Typography, 
  Paper, 
  Collapse 
} from "@mui/material";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ExpandLessIcon from "@mui/icons-material/ExpandLess";
import type { ReactionType } from "./MapComponent";

const REACTIONS: { type: ReactionType; emoji: string; label: string }[] = [
  { type: "icy",        emoji: "❄️", label: "Icy"        },
  { type: "powder",     emoji: "⛷️", label: "Powder"     },
  { type: "bluebird",   emoji: "☀️", label: "Bluebird"   },
  { type: "crowded",    emoji: "👥", label: "Crowded"    },
  { type: "heavy_snow", emoji: "🌨️", label: "Heavy Snow" },
  { type: "foggy",      emoji: "🌫️", label: "Foggy"      },
  { type: "sketchy",    emoji: "⚠️", label: "Sketchy"    },
  { type: "avalanche",  emoji: "🏔️", label: "Avalanche"  },
];

interface SidebarProps {
  onSubmit: (coords: { lat: number; lng: number }) => void;
  sendReaction: (reactionType: ReactionType, message: string) => void;
  pendingLocation: { lat: number; lng: number } | null;
  isLoggedIn: boolean;
}

export default function Sidebar({ onSubmit, sendReaction, pendingLocation }: SidebarProps) {
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

  const canPost = selectedType !== null && pendingLocation !== null;

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
      width: 280, // Slightly widened to accommodate grid layout better
      flexShrink: 0,
      bgcolor: "#0a1628",
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 3,
      p: 2.5,
      display: "flex",
      flexDirection: "column",
      gap: 2.5,
      mt: "40px",
      overflowY: "auto",
    }}>

      {/* ── Drop a Reaction (Now the primary focus) ── */}
      <Box>
        <Typography variant="h6" fontWeight={700} color="#fff" mb={0.5}>
          Current Conditions
        </Typography>
        <Typography variant="body2" color="rgba(255,255,255,0.6)" mb={2}>
          Click the map to select a location, then drop a reaction below.
        </Typography>

        <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>

          {/* Pinned location indicator - moved up for logical flow */}
          <Box sx={{ bgcolor: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 2, p: 1.5 }}>
            <Typography variant="caption" color="rgba(255,255,255,0.4)" display="block" mb={0.5} textTransform="uppercase" letterSpacing={0.5} fontSize={10}>
              Selected Location
            </Typography>
            {pendingLocation ? (
              <Typography variant="body2" color="#fff" fontFamily="monospace" fontWeight={500}>
                {pendingLocation.lat.toFixed(4)}, {pendingLocation.lng.toFixed(4)}
              </Typography>
            ) : (
              <Typography variant="body2" color="rgba(255,255,255,0.4)" fontStyle="italic">
                No location selected
              </Typography>
            )}
          </Box>

          {/* Emoji type selector using CSS Grid */}
          <Box sx={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 1.25
          }}>
            {REACTIONS.map(({ type, emoji, label }) => (
              <Button
                key={type}
                variant="outlined"
                onClick={() => setSelectedType(prev => prev === type ? null : type)}
                sx={{
                  flexDirection: "column",
                  textTransform: "none",
                  fontSize: 12,
                  p: 1.5,
                  lineHeight: 1.2,
                  gap: 0.5,
                  borderRadius: 2,
                  borderColor: selectedType === type
                    ? "rgba(255,255,255,0.8)"
                    : "rgba(255,255,255,0.12)",
                  color: selectedType === type ? "#fff" : "rgba(255,255,255,0.6)",
                  bgcolor: selectedType === type ? "rgba(255,255,255,0.1)" : "transparent",
                  transition: "all 0.2s ease-in-out",
                  "&:hover": {
                    borderColor: "rgba(255,255,255,0.5)",
                    bgcolor: "rgba(255,255,255,0.06)",
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
            onChange={e => setMessage(e.target.value)}
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
      </Box>

      <Divider sx={{ borderColor: "rgba(255,255,255,0.08)" }} />

      {/* ── Collapsible Location Search ── */}
      <Box>
        <Button 
          fullWidth 
          onClick={() => setShowManualEntry(!showManualEntry)}
          endIcon={showManualEntry ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          sx={{ 
            color: "rgba(255,255,255,0.5)", 
            textTransform: "none",
            justifyContent: "space-between",
            px: 1,
            "&:hover": { bgcolor: "rgba(255,255,255,0.05)", color: "#fff" }
          }}
        >
          Enter coordinates manually
        </Button>

        <Collapse in={showManualEntry}>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5, mt: 2, p: 1 }}>
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
                color: "rgba(255,255,255,0.7)",
                borderColor: "rgba(255,255,255,0.2)",
                borderRadius: 2,
                "&:hover": {
                  borderColor: "rgba(255,255,255,0.5)",
                  color: "#fff",
                  bgcolor: "rgba(255,255,255,0.05)",
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