import './App.css'
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import { AppBar, Toolbar, Typography, Box, Avatar, IconButton, Tooltip } from "@mui/material";
import MapIcon from "@mui/icons-material/Map";
import Home from "./pages/Home";
import Account from "./pages/Account";
import Map from "./pages/Map";


function Navbar() {
  return (
    <AppBar
      position="static"
      elevation={0}
      sx={{
        bgcolor: "#0a1628",
        borderBottom: "1px solid rgba(255,255,255,0.08)",
      }}
    >
      <Toolbar sx={{ gap: 1, px: { xs: 2, md: 4 } }}>

        {/* Logo — clicks to home */}
        <Box
          component={Link}
          to="/"
          sx={{
            display: "flex", alignItems: "center", gap: 1.5,
            textDecoration: "none", flexGrow: 1,
          }}
        >
          <Box sx={{
            bgcolor: "#1565c0",
            borderRadius: 2,
            width: 34, height: 34,
            display: "flex", alignItems: "center", justifyContent: "center"
          }}>
            <MapIcon sx={{ color: "#fff", fontSize: 20 }} />
          </Box>
          <Box>
            <Typography fontWeight={800} color="#fff" fontSize="0.95rem"
              sx={{ letterSpacing: "-0.02em"}}>
              AvalancheWatch
            </Typography>
          </Box>
        </Box>

        {/* Right side icons */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Tooltip title="Account">
            <IconButton component={Link} to="/account" sx={{ p: 0.5 }}>
              <Avatar sx={{ width: 34, height: 34, bgcolor: "#7b1fa2", fontSize: "0.85rem", fontWeight: 700 }}>
                JD
              </Avatar>
            </IconButton>
          </Tooltip>
        </Box>

      </Toolbar>
    </AppBar>
  );
}


export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/account" element={<Account />} />
        <Route path="/map" element={<Map />} />
      </Routes>
    </BrowserRouter>
  );
}
