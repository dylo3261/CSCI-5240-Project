import { useState } from "react";
import { Box, Typography } from "@mui/material";
import MapComponent from "../components/MapComponent";
import Sidebar from "../components/Sidebar";

const getColor = (v: number) => `hsl(${(1 - v) * 240}, 90%, 50%)`;


//function for the map page
export default function Map() {
  //placeholder zones for shading the map
  const [submittedCoords, setSubmittedCoords] = useState<{lat: number, lng: number} | null>(null);

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
      <Sidebar onSubmit={setSubmittedCoords} />
    </Box>
  );
}