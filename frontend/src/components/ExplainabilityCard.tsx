import { useEffect, useState } from "react";
import { Box, Typography, IconButton, Skeleton, Chip } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";

// ─────────────────────────────────────────────
// 🔌 PLUG IN YOUR ENDPOINT HERE
// Replace with your actual explainability API URL.
// It should accept POST with { latitude, longitude }
// and return { summary, factors, confidence, ... }
// ─────────────────────────────────────────────
const EXPLAINABILITY_API_URL = "https://YOUR-API-ENDPOINT/explain";

// Shape of the response your model returns.
// Adjust this to match your actual API response schema.
interface ExplainabilityResult {
  summary: string;           // e.g. "High avalanche risk due to recent snowfall"
  factors: {
    label: string;           // e.g. "Snowpack depth"
    value: string;           // e.g. "142 cm"
    contribution: "high" | "medium" | "low"; // impact on the prediction
  }[];
  confidence: number;        // 0–1
  riskLevel: "low" | "moderate" | "high" | "extreme";
}

// ─── Example/fallback data shown when API is not yet connected ───
const EXAMPLE_DATA: ExplainabilityResult = {
  summary:
    "Elevated avalanche risk in this zone based on recent precipitation patterns and terrain slope analysis.",
  factors: [
    { label: "Recent Snowfall", value: "38 cm (48h)", contribution: "high" },
    { label: "Slope Angle", value: "34°", contribution: "high" },
    { label: "Wind Loading", value: "Moderate NW", contribution: "medium" },
    { label: "Aspect", value: "NE-facing", contribution: "medium" },
    { label: "Elevation", value: "3,420 m", contribution: "low" },
  ],
  confidence: 0.81,
  riskLevel: "high",
};

const RISK_CONFIG = {
  low:      { color: "#4caf50", label: "Low Risk",      bg: "rgba(76,175,80,0.12)"   },
  moderate: { color: "#ff9800", label: "Moderate Risk", bg: "rgba(255,152,0,0.12)"   },
  high:     { color: "#f44336", label: "High Risk",     bg: "rgba(244,67,54,0.12)"   },
  extreme:  { color: "#b71c1c", label: "Extreme Risk",  bg: "rgba(183,28,28,0.18)"   },
};

const CONTRIBUTION_COLOR = {
  high:   "#f44336",
  medium: "#ff9800",
  low:    "#4caf50",
};

interface Props {
  location: { lat: number; lng: number } | null;
  onClose: () => void;
}

export default function ExplainabilityCard({ location, onClose }: Props) {
  const [result, setResult] = useState<ExplainabilityResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (!location) {
      setVisible(false);
      return;
    }

    setResult(null);
    setError(null);
    setLoading(true);

    // Trigger entrance animation after mount
    const animTimer = setTimeout(() => setVisible(true), 10);

    const controller = new AbortController();

    fetch(EXPLAINABILITY_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ latitude: location.lat, longitude: location.lng }),
      signal: controller.signal,
    })
      .then((r) => {
        if (!r.ok) throw new Error(`API error ${r.status}`);
        return r.json() as Promise<ExplainabilityResult>;
      })
      .then((data) => {
        setResult(data);
        setLoading(false);
      })
      .catch((err) => {
        if (err.name === "AbortError") return;
        console.warn("Explainability API not reachable, showing example data.", err);
        // ── Falls back to example data while your API is being wired up ──
        setResult(EXAMPLE_DATA);
        setLoading(false);
      });

    return () => {
      controller.abort();
      clearTimeout(animTimer);
    };
  }, [location]);

  if (!location) return null;

  const risk = result ? RISK_CONFIG[result.riskLevel] : null;

  return (
    <Box
      sx={{
        position: "absolute",
        // ── Positioning: sits over the map, top-right corner ──
        top: 16,
        right: 16,
        zIndex: 1000,
        width: 300,
        bgcolor: "#0a1628",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 3,
        overflow: "hidden",
        boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
        // Slide-in animation
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(-12px)",
        transition: "opacity 0.25s ease, transform 0.25s ease",
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          px: 2,
          py: 1.5,
          borderBottom: "1px solid rgba(255,255,255,0.07)",
          bgcolor: "rgba(255,255,255,0.03)",
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <InfoOutlinedIcon sx={{ color: "rgba(255,255,255,0.5)", fontSize: 16 }} />
          <Typography variant="caption" fontWeight={700} color="rgba(255,255,255,0.7)" textTransform="uppercase" letterSpacing={0.8} fontSize={10}>
            Model Explainability
          </Typography>
        </Box>
        <IconButton size="small" onClick={onClose} sx={{ color: "rgba(255,255,255,0.4)", p: 0.25, "&:hover": { color: "#fff" } }}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* Coordinates */}
      <Box sx={{ px: 2, pt: 1.5, pb: 1 }}>
        <Typography variant="caption" color="rgba(255,255,255,0.35)" fontFamily="monospace" fontSize={10}>
          {location.lat.toFixed(5)}, {location.lng.toFixed(5)}
        </Typography>
      </Box>

      {/* Body */}
      <Box sx={{ px: 2, pb: 2, display: "flex", flexDirection: "column", gap: 1.75 }}>

        {/* Risk badge + confidence */}
        {loading ? (
          <>
            <Skeleton variant="rounded" width={110} height={28} sx={{ bgcolor: "rgba(255,255,255,0.06)" }} />
            <Skeleton variant="text" width="90%" sx={{ bgcolor: "rgba(255,255,255,0.06)" }} />
            <Skeleton variant="text" width="75%" sx={{ bgcolor: "rgba(255,255,255,0.06)" }} />
          </>
        ) : result && risk ? (
          <>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
              <Chip
                label={risk.label}
                size="small"
                sx={{
                  bgcolor: risk.bg,
                  color: risk.color,
                  fontWeight: 700,
                  fontSize: 11,
                  border: `1px solid ${risk.color}40`,
                  height: 24,
                }}
              />
              <Typography variant="caption" color="rgba(255,255,255,0.4)" fontSize={11}>
                {Math.round(result.confidence * 100)}% confidence
              </Typography>
            </Box>

            {/* Summary */}
            <Typography variant="body2" color="rgba(255,255,255,0.75)" lineHeight={1.55} fontSize={12.5}>
              {result.summary}
            </Typography>

            {/* Contributing factors */}
            <Box>
              <Typography variant="caption" color="rgba(255,255,255,0.35)" textTransform="uppercase" letterSpacing={0.6} fontSize={10} display="block" mb={1}>
                Contributing Factors
              </Typography>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 0.75 }}>
                {result.factors.map((f) => (
                  <Box
                    key={f.label}
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      bgcolor: "rgba(255,255,255,0.03)",
                      border: "1px solid rgba(255,255,255,0.05)",
                      borderRadius: 1.5,
                      px: 1.25,
                      py: 0.75,
                    }}
                  >
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      {/* Contribution dot */}
                      <Box
                        sx={{
                          width: 6,
                          height: 6,
                          borderRadius: "50%",
                          bgcolor: CONTRIBUTION_COLOR[f.contribution],
                          flexShrink: 0,
                        }}
                      />
                      <Typography variant="caption" color="rgba(255,255,255,0.6)" fontSize={12}>
                        {f.label}
                      </Typography>
                    </Box>
                    <Typography variant="caption" color="#fff" fontWeight={600} fontSize={12} fontFamily="monospace">
                      {f.value}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </Box>

            {/* Legend */}
            <Box sx={{ display: "flex", gap: 1.5, flexWrap: "wrap" }}>
              {(["high", "medium", "low"] as const).map((level) => (
                <Box key={level} sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                  <Box sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: CONTRIBUTION_COLOR[level] }} />
                  <Typography variant="caption" color="rgba(255,255,255,0.3)" fontSize={10} textTransform="capitalize">
                    {level}
                  </Typography>
                </Box>
              ))}
            </Box>
          </>
        ) : error ? (
          <Typography variant="body2" color="rgba(255,100,100,0.8)" fontSize={12}>
            {error}
          </Typography>
        ) : null}
      </Box>
    </Box>
  );
}