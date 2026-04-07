import './App.css';
import { Amplify } from 'aws-amplify';
import { awsConfig } from './aws-config';
import '@aws-amplify/ui-react/styles.css';

Amplify.configure(awsConfig);
import { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import { AppBar, Toolbar, Typography, Box, Avatar, IconButton, Tooltip, createTheme, ThemeProvider } from "@mui/material";
import AcUnitIcon from "@mui/icons-material/AcUnit";
import { fetchUserAttributes } from "aws-amplify/auth";
import Account from "./pages/Account";
import Map from "./pages/Map";

const slabLabTheme = createTheme({
  typography: {
    fontFamily: "'Space Grotesk', system-ui, sans-serif",
  },
  palette: {
    mode: "dark",
    background: { default: "#0D0A1A", paper: "#1E1432" },
    primary: { main: "#FF2D78" },
  },
});

function Navbar() {
  const [initials, setInitials] = useState("--");

  useEffect(() => {
    fetchUserAttributes()
      .then((attrs) => {
        const name = attrs.name || attrs.email || "";
        const parts = name.trim().split(/\s+/);
        if (parts.length >= 2) {
          setInitials((parts[0][0] + parts[parts.length - 1][0]).toUpperCase());
        } else if (parts[0]?.length) {
          setInitials(parts[0].slice(0, 2).toUpperCase());
        }
      })
      .catch(() => {
        // Not authenticated — keep "--"
      });
  }, []);

  return (
    <AppBar
      position="static"
      elevation={0}
      sx={{
        bgcolor: "#0D0A1A",
        borderBottom: "1px solid rgba(240,248,255,0.08)",
        backgroundImage: "none",
      }}
    >
      <Toolbar sx={{ gap: 1, px: { xs: 2, md: 4 }, minHeight: "48px !important" }}>

        {/* Logo */}
        <Box
          component={Link}
          to="/"
          sx={{ display: "flex", alignItems: "center", gap: 1.5, textDecoration: "none", flexGrow: 1 }}
        >
          <Box sx={{
            background: "linear-gradient(135deg, #FF2D78 0%, #BF5FFF 100%)",
            boxShadow: "0 0 10px rgba(255,45,120,0.45)",
            borderRadius: 1.5,
            width: 30, height: 30,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <AcUnitIcon sx={{ color: "#fff", fontSize: 18 }} />
          </Box>
          <Box>
            <Typography
              sx={{ fontFamily: "'Righteous', sans-serif", letterSpacing: "0.05em", lineHeight: 1, fontSize: "1.1rem", color: "#F0F8FF" }}
            >
              SLAB LAB
            </Typography>
          </Box>
        </Box>

        {/* Account avatar */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Tooltip title="Account">
            <IconButton component={Link} to="/account" sx={{ p: 0.5 }}>
              <Avatar sx={{
                width: 30, height: 30,
                background: "linear-gradient(135deg, #00E5CC 0%, #0088FF 100%)",
                fontSize: "0.75rem", fontWeight: 700,
                boxShadow: "0 0 8px rgba(0,229,204,0.35)",
                fontFamily: "'Space Grotesk', sans-serif",
              }}>
                {initials}
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
    <ThemeProvider theme={slabLabTheme}>
      <BrowserRouter>
        <Navbar />
        <Routes>
          <Route path="/" element={<Map />} />
          <Route path="/map" element={<Map />} />
          <Route path="/account" element={<Account />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}
