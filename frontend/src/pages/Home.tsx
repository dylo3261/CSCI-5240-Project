import { Box, Typography, Chip, Button, Card, CardActionArea, CardContent } from "@mui/material";
import AcUnitIcon from "@mui/icons-material/AcUnit";
import TerrainIcon from "@mui/icons-material/Terrain";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();

  return (
    <Box sx={{ bgcolor: "#0D0A1A", minHeight: "calc(100vh - 64px)" }}>

      {/* ── Hero ── */}
      <Box sx={{
        position: "relative",
        minHeight: 460,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        px: 3,
        py: 8,
        overflow: "hidden",
        background: `
          linear-gradient(to bottom, rgba(13,10,26,0.4) 0%, rgba(13,10,26,0.92) 100%),
          url('https://images.unsplash.com/photo-1551524559-8af4e6624178?w=1600&q=80') center/cover no-repeat
        `,
      }}>
        {/* Zigzag geometric accent — top-left */}
        <Box sx={{
          position: "absolute", top: 24, left: 24, opacity: 0.18,
          display: { xs: "none", md: "block" },
        }}>
          <svg width="80" height="80" viewBox="0 0 80 80">
            <polyline points="0,20 20,0 40,20 60,0 80,20" fill="none" stroke="#FF2D78" strokeWidth="3"/>
            <polyline points="0,40 20,20 40,40 60,20 80,40" fill="none" stroke="#00E5CC" strokeWidth="3"/>
            <polyline points="0,60 20,40 40,60 60,40 80,60" fill="none" stroke="#BF5FFF" strokeWidth="3"/>
          </svg>
        </Box>

        {/* Zigzag geometric accent — top-right */}
        <Box sx={{
          position: "absolute", top: 24, right: 24, opacity: 0.18,
          display: { xs: "none", md: "block" },
        }}>
          <svg width="80" height="80" viewBox="0 0 80 80">
            <polyline points="0,20 20,0 40,20 60,0 80,20" fill="none" stroke="#FFAA00" strokeWidth="3"/>
            <polyline points="0,40 20,20 40,40 60,20 80,40" fill="none" stroke="#0088FF" strokeWidth="3"/>
            <polyline points="0,60 20,40 40,60 60,40 80,60" fill="none" stroke="#FF6B35" strokeWidth="3"/>
          </svg>
        </Box>

        {/* Retro badge */}
        <Box sx={{
          mb: 2,
          display: "inline-flex",
          alignItems: "center",
          gap: 0.75,
          border: "1.5px solid rgba(0,229,204,0.5)",
          borderRadius: "100px",
          px: 2, py: 0.5,
          bgcolor: "rgba(0,229,204,0.08)",
        }}>
          <AcUnitIcon sx={{ color: "#00E5CC", fontSize: 13 }} />
          <Typography fontSize="0.65rem" letterSpacing="0.15em" textTransform="uppercase" color="#00E5CC" fontWeight={600}>
            Colorado Backcountry Intelligence
          </Typography>
        </Box>

        {/* Main title */}
        <Typography
          variant="h1"
          sx={{
            fontFamily: "'Righteous', sans-serif",
            fontSize: { xs: "3.5rem", md: "6rem" },
            letterSpacing: "0.05em",
            lineHeight: 1,
            mb: 1,
            background: "linear-gradient(135deg, #FF2D78 0%, #BF5FFF 50%, #00D4FF 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
            filter: "drop-shadow(0 0 30px rgba(255,45,120,0.35))",
          }}
        >
          SLAB LAB
        </Typography>

        <Typography
          variant="body1"
          sx={{ color: "rgba(240,248,255,0.65)", maxWidth: 480, mb: 4, lineHeight: 1.7, fontSize: "1rem" }}
        >
          AI-powered avalanche risk for Colorado's backcountry. Click the map, get the read — stay safe out there.
        </Typography>

        <Button
          onClick={() => navigate("/map")}
          variant="contained"
          startIcon={<TerrainIcon />}
          sx={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontWeight: 700,
            fontSize: "0.95rem",
            textTransform: "none",
            px: 4, py: 1.5,
            borderRadius: "100px",
            background: "linear-gradient(135deg, #FF2D78 0%, #BF5FFF 100%)",
            boxShadow: "0 0 24px rgba(255,45,120,0.45)",
            "&:hover": {
              background: "linear-gradient(135deg, #FF2D78 0%, #0088FF 100%)",
              boxShadow: "0 0 32px rgba(255,45,120,0.6)",
            },
          }}
        >
          Enter the Mountain
        </Button>

        {/* Mountain silhouette SVG at bottom of hero */}
        <Box sx={{ position: "absolute", bottom: 0, left: 0, right: 0, zIndex: 1, lineHeight: 0 }}>
          <svg viewBox="0 0 1440 100" style={{ width: "100%", display: "block" }} preserveAspectRatio="none">
            <polygon
              points="0,100 0,70 160,22 320,58 520,8 700,42 880,12 1080,48 1260,18 1440,35 1440,100"
              fill="#0D0A1A"
            />
          </svg>
        </Box>
      </Box>

      {/* ── Stat bar ── */}
      <Box sx={{
        display: "flex",
        justifyContent: "center",
        gap: { xs: 4, md: 8 },
        py: 3,
        px: 3,
        borderBottom: "1px solid rgba(240,248,255,0.06)",
      }}>
        {[
          { value: "118", label: "SNOTEL Stations" },
          { value: "9.5K+", label: "Avalanche Events" },
          { value: "RandomForest", label: "ML Model" },
        ].map(({ value, label }) => (
          <Box key={label} sx={{ textAlign: "center" }}>
            <Typography
              sx={{ fontFamily: "'Righteous', sans-serif", fontSize: "1.4rem", color: "#00E5CC", lineHeight: 1 }}
            >
              {value}
            </Typography>
            <Typography variant="caption" color="rgba(240,248,255,0.4)" letterSpacing="0.08em" textTransform="uppercase">
              {label}
            </Typography>
          </Box>
        ))}
      </Box>

      {/* ── Map Section ── */}
      <Box sx={{ px: { xs: 2, md: 6 }, py: 5 }}>
        <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", mb: 3 }}>
          <Box>
            <Typography
              variant="h5"
              sx={{ fontFamily: "'Righteous', sans-serif", color: "#F0F8FF", letterSpacing: "0.03em" }}
            >
              Live Risk Map
            </Typography>
            <Typography variant="body2" color="rgba(240,248,255,0.45)" mt={0.5}>
              Real-time avalanche risk zones across Colorado
            </Typography>
          </Box>
          <Button
            onClick={() => navigate("/map")}
            endIcon={<TerrainIcon />}
            sx={{
              color: "rgba(240,248,255,0.55)",
              textTransform: "none",
              fontFamily: "'Space Grotesk', sans-serif",
              "&:hover": { color: "#00E5CC" },
            }}
          >
            Full Map
          </Button>
        </Box>

        {/* Clickable map card */}
        <Card sx={{
          borderRadius: 3, overflow: "hidden",
          border: "1px solid rgba(240,248,255,0.08)",
          bgcolor: "#1E1432",
          transition: "border-color 0.2s, box-shadow 0.2s",
          "&:hover": {
            borderColor: "rgba(255,45,120,0.4)",
            boxShadow: "0 0 32px rgba(255,45,120,0.12)",
          },
        }} elevation={0}>
          <CardActionArea onClick={() => navigate("/map")}>

            <Box sx={{ height: { xs: 280, md: 460 }, pointerEvents: "none" }}>
              <img src="/map-placeholder.png" style={{ height: "100%", width: "100%", objectFit: "cover", borderRadius: "12px 12px 0 0" }} />
            </Box>

            <CardContent sx={{
              display: "flex", alignItems: "center",
              justifyContent: "space-between",
              bgcolor: "#1E1432", px: 3,
            }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
                <AcUnitIcon sx={{ color: "#FF2D78", filter: "drop-shadow(0 0 6px rgba(255,45,120,0.7))" }} />
                <Box>
                  <Typography fontWeight={700} color="#F0F8FF" fontFamily="'Space Grotesk', sans-serif">
                    View Full Interactive Map
                  </Typography>
                  <Typography variant="caption" color="rgba(240,248,255,0.4)">
                    Click to explore risk zones in detail
                  </Typography>
                </Box>
              </Box>
              <Box sx={{ display: "flex", gap: 1 }}>
                <Chip label="High" size="small"
                  sx={{ bgcolor: "rgba(255,45,120,0.15)", color: "#FF2D78", border: "1px solid rgba(255,45,120,0.4)", fontSize: "0.7rem", fontFamily: "'Space Grotesk', sans-serif" }} />
                <Chip label="Medium" size="small"
                  sx={{ bgcolor: "rgba(255,170,0,0.15)", color: "#FFAA00", border: "1px solid rgba(255,170,0,0.4)", fontSize: "0.7rem", fontFamily: "'Space Grotesk', sans-serif" }} />
                <Chip label="Low" size="small"
                  sx={{ bgcolor: "rgba(0,229,204,0.12)", color: "#00E5CC", border: "1px solid rgba(0,229,204,0.4)", fontSize: "0.7rem", fontFamily: "'Space Grotesk', sans-serif" }} />
              </Box>
            </CardContent>

          </CardActionArea>
        </Card>
      </Box>

    </Box>
  );
}
