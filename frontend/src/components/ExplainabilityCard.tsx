import { useEffect, useState } from "react";
import { Box, Typography, IconButton, Skeleton, Chip, Collapse, CircularProgress } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import DownloadIcon from "@mui/icons-material/Download";

const EXPLAINABILITY_API_URL =
  "https://mera3wkzuj.execute-api.us-west-2.amazonaws.com/request-redirector";

// ── Card display format ────────────────────────────────────────────────────
interface ExplainabilityResult {
  summary: string;
  factors: {
    label: string;
    value: string;
    contribution: "high" | "medium" | "low";
    direction: "positive" | "negative" | "neutral";
  }[];
  confidence: number;
  riskLevel: "low" | "moderate" | "considerable" | "high" | "extreme";
  terrain: { elevation: number; slope: number; aspect_degrees: number };
  weather: { snow_depth: number; new_snow_24h: number; swe: number; temp: number; snow_ratio: number };
  stations_used: { station_id: number; station_triplet: string; name: string; distance_km: number }[];
}

// ── Raw response from explainable-inference Lambda ─────────────────────────
interface RawExplanationItem {
  feature: string;
  value: number;
  shap_impact: number;
  direction: string;
  context: string | null;
}

interface RawInferenceResponse {
  prediction: number;
  risk_level: string;
  explanation: RawExplanationItem[];
  terrain: { elevation: number; slope: number; aspect_degrees: number };
  weather: { snow_depth: number; new_snow_24h: number; swe: number; temp: number; snow_ratio: number };
  stations_used: { station_id: number; station_triplet: string; name: string; distance_km: number }[];
  base_value?: number;
}

// ── Feature display config ─────────────────────────────────────────────────
const FEATURE_LABEL: Record<string, string> = {
  elevation:      "Elevation",
  slope:          "Slope Angle",
  aspect_degrees: "Aspect",
  snow_depth:     "Snow Depth",
  new_snow_24h:   "New Snow (24h)",
  temp:           "Temperature",
  snow_ratio:     "Snow Ratio",
};

function formatFeatureValue(feature: string, value: number): string {
  switch (feature) {
    case "elevation":      return `${Math.round(value)} m`;
    case "slope":          return `${value.toFixed(1)}°`;
    case "aspect_degrees": return `${Math.round(value)}°`;
    case "snow_depth":     return `${value.toFixed(1)} cm`;
    case "new_snow_24h":   return `${value.toFixed(1)} cm`;
    case "temp":           return `${value.toFixed(1)}°C`;
    case "snow_ratio":     return value.toFixed(2);
    default:               return String(value);
  }
}

function shapContribution(impact: number): "high" | "medium" | "low" {
  const abs = Math.abs(impact);
  if (abs > 0.08) return "high";
  if (abs > 0.02) return "medium";
  return "low";
}

function transformResponse(raw: RawInferenceResponse): ExplainabilityResult {
  const riskLevel = raw.risk_level.toLowerCase() as ExplainabilityResult["riskLevel"];

  const topFactor = raw.explanation[0];
  const contextHint = topFactor?.context ? ` — ${topFactor.context}` : "";
  const summary = `${raw.risk_level} avalanche risk${contextHint}.`;

  const factors = raw.explanation.map((item) => ({
    label: FEATURE_LABEL[item.feature] ?? item.feature,
    value: formatFeatureValue(item.feature, item.value),
    contribution: shapContribution(item.shap_impact),
    direction: (
      item.direction === "positive" || item.direction === "increasing" ? "positive"
      : item.direction === "negative" || item.direction === "decreasing" ? "negative"
      : "neutral"
    ) as "positive" | "negative" | "neutral",
  }));

  return {
    summary,
    factors,
    confidence: raw.prediction,
    riskLevel,
    terrain: raw.terrain,
    weather: raw.weather,
    stations_used: raw.stations_used ?? [],
  };
}

// ── Risk level display config ──────────────────────────────────────────────
const RISK_CONFIG = {
  low:          { color: "#4caf50", label: "Low Risk",          bg: "rgba(76,175,80,0.12)"   },
  moderate:     { color: "#ff9800", label: "Moderate Risk",     bg: "rgba(255,152,0,0.12)"   },
  considerable: { color: "#ff6f00", label: "Considerable Risk", bg: "rgba(255,111,0,0.12)"   },
  high:         { color: "#f44336", label: "High Risk",         bg: "rgba(244,67,54,0.12)"   },
  extreme:      { color: "#b71c1c", label: "Extreme Risk",      bg: "rgba(183,28,28,0.18)"   },
};


// ── Reusable collapsible section header ────────────────────────────────────
function SectionHeader({ label, open, onToggle }: { label: string; open: boolean; onToggle: () => void }) {
  return (
    <Box
      onClick={onToggle}
      sx={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        cursor: "pointer",
        py: 0.5,
        "&:hover": { opacity: 0.75 },
      }}
    >
      <Typography variant="caption" color="rgba(255,255,255,0.35)" textTransform="uppercase" letterSpacing={0.6} fontSize={10}>
        {label}
      </Typography>
      <ExpandMoreIcon
        sx={{
          color: "rgba(255,255,255,0.3)",
          fontSize: 14,
          transition: "transform 0.2s",
          transform: open ? "rotate(180deg)" : "rotate(0deg)",
        }}
      />
    </Box>
  );
}


interface Props {
  location: { lat: number; lng: number } | null;
  open: boolean;
  onClose: () => void;
}

export default function ExplainabilityCard({ location, open, onClose }: Props) {
  const [result, setResult] = useState<ExplainabilityResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [terrainOpen, setTerrainOpen] = useState(false);
  const [weatherOpen, setWeatherOpen] = useState(false);

useEffect(() => {
    if (!location) {
      return;
    }

    setResult(null);
    setError(null);
    setLoading(true);
    setTerrainOpen(false);
    setWeatherOpen(false);

    const controller = new AbortController();

    fetch(EXPLAINABILITY_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ latitude: location.lat, longitude: location.lng }),
      signal: controller.signal,
    })
      .then((r) => r.json())
      .then((outer) => {
        // API Gateway unwraps the redirector envelope — outer IS the inner Lambda response
        // shape: { statusCode, headers, body: "<JSON string of actual data>" }
        if (outer.statusCode && outer.statusCode !== 200) {
          const errBody = JSON.parse(outer.body as string);
          const raw: string = errBody.error ?? `Error ${outer.statusCode}`;
          const msg = raw.toLowerCase().includes("outside")
            ? "Prediction not supported for this location."
            : raw.toLowerCase().includes("missing feature") || raw.toLowerCase().includes("snotel")
            ? "No weather station data available near this location."
            : "Unable to generate a prediction for this location.";
          throw new Error(msg);
        }
        return JSON.parse(outer.body as string) as RawInferenceResponse;
      })
      .then((raw) => {
        setResult(transformResponse(raw));
        setLoading(false);
      })
      .catch((err: Error) => {
        if (err.name === "AbortError") return;
        setError(err.message ?? "Failed to load prediction.");
        setLoading(false);
      });

    return () => {
      controller.abort();
    };
  }, [location]);

  if (!location || !open) return null;

  const risk = result ? RISK_CONFIG[result.riskLevel] : null;

  function handleDownload() {
    if (!result || !location) return;
    const directionSymbol = (d: string) => d === "positive" ? "↑" : d === "negative" ? "↓" : " ";
    const lines = [
      "SLAB LAB — RISK REPORT",
      "=======================",
      `Date:       ${new Date().toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" })}`,
      `Location:   ${location.lat.toFixed(5)}, ${location.lng.toFixed(5)}`,
      "",
      `Risk Level: ${RISK_CONFIG[result.riskLevel].label.toUpperCase()}`,
      `Confidence: ${Math.round(result.confidence * 100)}%`,
      "",
      result.summary,
      "",
      "CONTRIBUTING FACTORS",
      "--------------------",
      ...result.factors.map((f) => `  ${directionSymbol(f.direction)} ${f.label.padEnd(18)} ${f.value}`),
      "",
      "TERRAIN",
      "-------",
      `  Elevation:  ${Math.round(result.terrain.elevation)} m`,
      `  Slope:      ${result.terrain.slope.toFixed(1)}°`,
      `  Aspect:     ${Math.round(result.terrain.aspect_degrees)}°`,
      "",
      ...(result.stations_used.length > 0 ? [
        "WEATHER STATIONS USED",
        "---------------------",
        ...result.stations_used.map((s) => `  ${(s.name || s.station_triplet).padEnd(24)} ${s.distance_km.toFixed(1)} km`),
        "",
      ] : []),
      "↑ raises risk   ↓ lowers risk",
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `slab-lab-risk-${location.lat.toFixed(3)}-${location.lng.toFixed(3)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <Box
      sx={{
        position: "absolute",
        top: 16,
        right: 16,
        zIndex: 1000,
        width: 310,
        bgcolor: "#150E2A",
        border: "1px solid rgba(255,45,120,0.2)",
        borderRadius: 3,
        boxShadow: "0 8px 32px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,45,120,0.05)",
        opacity: 0,
        transform: "translateY(-12px)",
        animation: "explainability-card-in 0.25s ease forwards",
        "@keyframes explainability-card-in": {
          from: {
            opacity: 0,
            transform: "translateY(-12px)",
          },
          to: {
            opacity: 1,
            transform: "translateY(0)",
          },
        },
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
          bgcolor: "rgba(255,45,120,0.04)",
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <InfoOutlinedIcon sx={{ color: "#FF2D78", fontSize: 16 }} />
          <Typography variant="caption" fontWeight={700} color="rgba(240,248,255,0.7)" textTransform="uppercase" letterSpacing={0.8} fontSize={10} fontFamily="'Righteous', sans-serif">
            Slab Analysis
          </Typography>
          {loading && <CircularProgress size={10} thickness={5} sx={{ color: "rgba(255,255,255,0.4)" }} />}
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          {result && (
            <IconButton size="small" onClick={handleDownload} sx={{ color: "rgba(255,255,255,0.4)", p: 0.25, "&:hover": { color: "#fff" } }}>
              <DownloadIcon sx={{ fontSize: 15 }} />
            </IconButton>
          )}
          <IconButton size="small" onClick={onClose} sx={{ color: "rgba(255,255,255,0.4)", p: 0.25, "&:hover": { color: "#fff" } }}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
      </Box>

      {/* Coordinates */}
      <Box sx={{ px: 2, pt: 1.5, pb: 1 }}>
        <Typography variant="caption" color="rgba(255,255,255,0.35)" fontFamily="monospace" fontSize={10}>
          {location.lat.toFixed(5)}, {location.lng.toFixed(5)}
        </Typography>
      </Box>

      {/* Body */}
      <Box sx={{ px: 2, pb: 2, display: "flex", flexDirection: "column", gap: 1.75, maxHeight: "70vh", overflowY: "auto", "&::-webkit-scrollbar": { width: 4 }, "&::-webkit-scrollbar-track": { bgcolor: "transparent" }, "&::-webkit-scrollbar-thumb": { bgcolor: "rgba(255,255,255,0.15)", borderRadius: 2 } }}>
        {loading ? (
          <>
            <Skeleton variant="rounded" width={110} height={28} sx={{ bgcolor: "rgba(255,255,255,0.06)" }} />
            <Skeleton variant="text" width="90%" sx={{ bgcolor: "rgba(255,255,255,0.06)" }} />
            <Skeleton variant="text" width="75%" sx={{ bgcolor: "rgba(255,255,255,0.06)" }} />
          </>
        ) : result && risk ? (
          <>
            {/* Risk chip + confidence */}
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
              <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
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
                    {/* Left: direction arrow + label */}
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                      {f.direction === "positive" && (
                        <Typography component="span" sx={{ color: "#f44336", fontSize: 13, lineHeight: 1, fontWeight: 700 }}>↑</Typography>
                      )}
                      {f.direction === "negative" && (
                        <Typography component="span" sx={{ color: "#4caf50", fontSize: 13, lineHeight: 1, fontWeight: 700 }}>↓</Typography>
                      )}
                      {f.direction === "neutral" && <Box sx={{ width: 9 }} />}
                      <Typography variant="caption" color="rgba(255,255,255,0.6)" fontSize={12}>
                        {f.label}
                      </Typography>
                    </Box>

                    {/* Right: value */}
                    <Typography variant="caption" color="#fff" fontWeight={600} fontSize={12} fontFamily="monospace">
                      {f.value}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </Box>

            {/* Legend */}
            <Box sx={{ display: "flex", gap: 1.5, flexWrap: "wrap" }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                <Typography component="span" sx={{ color: "#f44336", fontSize: 13, lineHeight: 1, fontWeight: 700 }}>↑</Typography>
                <Typography variant="caption" color="rgba(255,255,255,0.3)" fontSize={10}>raises risk</Typography>
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                <Typography component="span" sx={{ color: "#4caf50", fontSize: 13, lineHeight: 1, fontWeight: 700 }}>↓</Typography>
                <Typography variant="caption" color="rgba(255,255,255,0.3)" fontSize={10}>lowers risk</Typography>
              </Box>
            </Box>

            {/* Terrain section */}
            <Box>
              <SectionHeader label="Terrain" open={terrainOpen} onToggle={() => setTerrainOpen((v) => !v)} />
              <Collapse in={terrainOpen}>
                <Box sx={{ pt: 0.5 }}>
                  {[
                    { label: "Elevation", value: `${Math.round(result.terrain.elevation)} m` },
                    { label: "Slope",     value: `${result.terrain.slope.toFixed(1)}°` },
                    { label: "Aspect",    value: `${Math.round(result.terrain.aspect_degrees)}°` },
                  ].map(({ label, value }) => (
                    <Box key={label} sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", py: 0.4 }}>
                      <Typography variant="caption" color="rgba(255,255,255,0.45)" fontSize={11}>{label}</Typography>
                      <Typography variant="caption" color="#fff" fontWeight={600} fontSize={11} fontFamily="monospace">{value}</Typography>
                    </Box>
                  ))}
                </Box>
              </Collapse>
            </Box>

            {/* Weather Stations section */}
            {result.stations_used.length > 0 && (
              <Box>
                <SectionHeader
                  label={`Weather Stations (${result.stations_used.length})`}
                  open={weatherOpen}
                  onToggle={() => setWeatherOpen((v) => !v)}
                />
                <Collapse in={weatherOpen}>
                  <Box sx={{ pt: 0.5 }}>
                    {result.stations_used.map((s) => (
                      <Box key={s.station_id} sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", py: 0.4 }}>
                        <Typography variant="caption" color="rgba(255,255,255,0.5)" fontSize={11} sx={{ maxWidth: 190, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {s.name || s.station_triplet}
                        </Typography>
                        <Typography variant="caption" color="rgba(255,255,255,0.3)" fontSize={11} fontFamily="monospace">
                          {s.distance_km.toFixed(1)} km
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                </Collapse>
              </Box>
            )}
          </>
        ) : error ? (
          <Typography variant="body2" color="rgba(255,255,255,0.4)" fontSize={12} lineHeight={1.5}>
            {error}
          </Typography>
        ) : null}
      </Box>
    </Box>
  );
}
