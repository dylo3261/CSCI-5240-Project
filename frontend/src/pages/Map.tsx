import { useState } from "react";
import { Button, TextField, Divider, Box, Typography } from "@mui/material";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import Paper from '@mui/material/Paper';
import MapComponent from "../components/MapComponent";

const getColor = (v: number) => `hsl(${(1 - v) * 240}, 90%, 50%)`;


//function for the map page
export default function Map() {
  //placeholder zones for shading the map
  const [lat, setLat] = useState("");
  const [lng, setLng] = useState("");

  const [submittedCoords, setSubmittedCoords] = useState<{lat: number, lng: number} | null>(null);

  //Submit function that sends the lat and lng to the backend API, and zooms in the map to those coordinates
  const handleSubmit = async () => {
    setSubmittedCoords({ lat: parseFloat(lat), lng: parseFloat(lng) });

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

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      console.log("API response:", data);
    } catch (err) {
      console.error("Failed to send coordinates:", err);
    }
  };

  return (
    <Box sx={{
      display: "flex",
      justifyContent: "center",
      alignItems: "flex-start",
      height: "calc(100vh - 64px)",
      bgcolor: "#0f1b2d",
      p: 3,
      }}>
          {/* Map */}
      <Box sx={{ flex: 1, height: "100%", minWidth: 0 }}>
        <Typography variant="h6" sx={{ color: "#fff", mb: 2 }}>
          Avalanche Risk Map
        </Typography>
        <Box sx={{ position: "relative", height: "calc(100% - 48px)", borderRadius: 3, overflow: "hidden" }}>
          <MapComponent coords={submittedCoords}/>

          {/* Legend overlay */}
        <Box sx={{
          position: "absolute",
          bottom: 24,
          right: 16,
          zIndex: 1000,
          bgcolor: "rgba(10, 22, 40, 0.85)",
          backdropFilter: "blur(8px)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 2,
          p: 1.5,
          minWidth: 160,
          }}>
          <Typography variant="caption" color="rgba(255,255,255,0.4)"
            display="block" mb={1} letterSpacing={0.5} textTransform="uppercase" fontSize={10}>
            Avalanche Risk
          </Typography>

          <Box sx={{
            height: 10,
            borderRadius: 1,
            mb: 0.75,
            background: `linear-gradient(to right, ${
              Array.from({ length: 10 }, (_, i) => getColor(i / 9)).join(", ")
            })`,
            }} />

            <Box sx={{ display: "flex", justifyContent: "space-between" }}>
              <Typography variant="caption" color="rgba(255,255,255,0.3)" fontSize={10}>Low</Typography>
              <Typography variant="caption" color="rgba(255,255,255,0.3)" fontSize={10}>High</Typography>
            </Box>
          </Box>
      </Box>
    </Box>

      {/* Sidebar */}
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
        mt: "40px", // aligns with map below the title
        }}>

        {/* Title */}
        <Box>
          <Typography variant="h6" fontWeight={700} color="#fff">
            Location Search
          </Typography>
        </Box>

        <Divider sx={{ borderColor: "rgba(255,255,255,0.08)" }} />

        {/* Lat input */}
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

        {/* Lng input */}
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

        {/* Locate Me button */}
        <Button
          onClick={handleSubmit}
          variant="outlined"
          startIcon={<MyLocationIcon />}
          fullWidth
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
          Locate Me
        </Button>

        <Divider sx={{ borderColor: "rgba(255,255,255,0.08)" }} />

        {/* Coords display */}
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
    </Box>
  );
}