import { Box, Typography, Chip, Button, Card, CardActionArea, CardContent } from "@mui/material";
import MapIcon from "@mui/icons-material/Map";
import { useNavigate } from "react-router-dom";
import MapComponent from "../components/MapComponent";

export default function Home() {
  const navigate = useNavigate();

  return (
    <Box sx={{ bgcolor: "#0f1b2d", minHeight: "calc(100vh - 64px)" }}>

      {/* ── Hero ── */}
      <Box sx={{
        position: "relative",
        minHeight: 420,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        px: 3,
        py: 8,
        overflow: "hidden",
        // mountain background overlay
        background: `linear-gradient(to bottom, rgba(10,20,40,0.55) 0%, rgba(10,20,40,0.85) 100%),
                     url('https://images.unsplash.com/photo-1551524559-8af4e6624178?w=1600&q=80') center/cover no-repeat`,
      }}>
        <Typography
          variant="h2"
          fontWeight={900}
          color="#fff"
          sx={{ letterSpacing: "-0.03em", lineHeight: 1.1, mb: 2,
            fontSize: { xs: "2.2rem", md: "3.5rem" } }}
        >
          Avalanche Watch
        </Typography>

        <Typography variant="body1" color="rgba(255,255,255,0.7)"
          sx={{ maxWidth: 520, mb: 4, lineHeight: 1.7 }}>
          Real-time avalanche risk detection and analysis for mountain regions across Colorado
        </Typography>
      </Box>

      {/* ── Map Section ── */}
      <Box sx={{ px: { xs: 2, md: 6 }, py: 5 }}>
        <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", mb: 3 }}>
          <Box>
            <Typography variant="h5" fontWeight={800} color="#fff">
              Risk Map
            </Typography>
            <Typography variant="body2" color="rgba(255,255,255,0.5)" mt={0.5}>
              Live avalanche risk zones across Colorado
            </Typography>
          </Box>
          <Button
            onClick={() => navigate("/map")}
            endIcon={<MapIcon />}
            sx={{ color: "rgba(255,255,255,0.6)", textTransform: "none",
              "&:hover": { color: "#fff" } }}
          >
            Full Map
          </Button>
        </Box>

        {/* Clickable map card */}
        <Card sx={{
          borderRadius: 3, overflow: "hidden",
          border: "1px solid rgba(255,255,255,0.08)",
          bgcolor: "#1a2a3a",
        }} elevation={6}>
          <CardActionArea onClick={() => navigate("/map")}>

            <Box sx={{ height: { xs: 280, md: 460 }, pointerEvents: "none" }}>
              <MapComponent />
            </Box>

            <CardContent sx={{
              display: "flex", alignItems: "center",
              justifyContent: "space-between",
              bgcolor: "#1a2a3a", px: 3
            }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
                <MapIcon sx={{ color: "#e8ff47" }} />
                <Box>
                  <Typography fontWeight={700} color="#fff">View Full Interactive Map</Typography>
                  <Typography variant="caption" color="rgba(255,255,255,0.45)">
                    Click to explore risk zones in detail
                  </Typography>
                </Box>
              </Box>
              <Box sx={{ display: "flex", gap: 1 }}>
                <Chip label="High" size="small"
                  sx={{ bgcolor: "#ff3b3b22", color: "#ff3b3b", border: "1px solid #ff3b3b44", fontSize: "0.7rem" }} />
                <Chip label="Medium" size="small"
                  sx={{ bgcolor: "#ff990022", color: "#ff9900", border: "1px solid #ff990044", fontSize: "0.7rem" }} />
                <Chip label="Low" size="small"
                  sx={{ bgcolor: "#ffe60022", color: "#ffe600", border: "1px solid #ffe60044", fontSize: "0.7rem" }} />
              </Box>
            </CardContent>

          </CardActionArea>
        </Card>
      </Box>

    </Box>
  );
}