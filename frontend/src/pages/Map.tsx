import { useState } from "react";
import { Box, Card, CardContent, Typography } from "@mui/material";
import MapComponent from "../components/MapComponent";

const PLACEHOLDER_ZONES = [
  {
    id: "zone_1",
    risk: "high",
    coordinates: [
      [46.85, -121.80],
      [46.88, -121.72],
      [46.83, -121.68],
      [46.79, -121.75],
    ],
  },
];

export default function Map() {
  const [zones, setZones] = useState(PLACEHOLDER_ZONES);

  return (
     <Box sx={{ height: "calc(100vh - 64px)", display: "flex", flexDirection: "column" }}>
      <Typography variant="h6" sx={{ p: 2 }}>
        Avalanche Risk Map
      </Typography>
      <Box sx={{ flex: 1 }}>
        <MapComponent />
      </Box>
    </Box>
  );
}