import { useState, useEffect, useRef, useCallback, Component, type ReactNode, type ErrorInfo } from "react";
import { Box, Typography } from "@mui/material";
import { fetchAuthSession, getCurrentUser } from "aws-amplify/auth";
import MapComponent, { type ReactionMarker, type ReactionType } from "../components/MapComponent";
import Sidebar from "../components/Sidebar";

// Runtime-validated set of valid reaction types — prevents unrecognized server
// values (e.g. wrong casing, renamed variants) from reaching the map renderer.
const VALID_REACTION_TYPES = new Set<string>([
  "icy", "powder", "bluebird", "crowded", "heavy_snow", "foggy", "sketchy", "avalanche",
]);

function isValidReactionType(value: string): value is ReactionType {
  return VALID_REACTION_TYPES.has(value);
}

// Error Boundary wrapping only MapComponent so a bad pin never unmounts Map
// (which would also close the WebSocket connection).
interface MapErrorBoundaryState { hasError: boolean; error: Error | null }
class MapErrorBoundary extends Component<{ children: ReactNode }, MapErrorBoundaryState> {
  state: MapErrorBoundaryState = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): MapErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("MapComponent crashed — recovering via error boundary:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Box sx={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", bgcolor: "#0f1b2d" }}>
          <Typography color="error" variant="body2">
            Map failed to render a pin. Reload the map to retry.
          </Typography>
        </Box>
      );
    }
    return this.props.children;
  }
}

// Replace with the wss:// URL from the SAM stack output WebSocketApiEndpoint
const WS_URL = "wss://j9jkzycsge.execute-api.us-west-2.amazonaws.com/prod";
// Replace with the https:// URL from the SAM stack output FetchReactionsEndpoint
const FETCH_REACTIONS_URL = "https://wpvtd43yyi.execute-api.us-west-2.amazonaws.com/reactions";

const getColor = (v: number) => `hsl(${(1 - v) * 120}, 90%, 45%)`;

export default function Map() {
  const [submittedCoords, setSubmittedCoords] = useState<{ lat: number; lng: number } | null>(null);
  const [reactions, setReactions] = useState<ReactionMarker[]>([]);
  const [pendingLocation, setPendingLocation] = useState<{ lat: number; lng: number } | null>(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userId, setUserId] = useState("");
  const wsRef = useRef<WebSocket | null>(null);

  // Load pins from the last 24 hours on mount
  useEffect(() => {
    fetch(FETCH_REACTIONS_URL)
      .then(r => {
        if (!r.ok) throw new Error(`Failed to fetch reactions: ${r.status}`);
        return r.json() as Promise<ReactionMarker[]>;
      })
      .then(data => setReactions(data.filter(r => isValidReactionType(r.reactionType))))
      .catch(err => console.error("Failed to load initial reactions:", err));
  }, []);

  useEffect(() => {
    let ws: WebSocket | undefined;
    let cancelled = false;

    (async () => {
      try {
        const session = await fetchAuthSession();
        if (cancelled || !session.tokens) return;

        const { userId: uid } = await getCurrentUser();
        if (cancelled) return;

        setUserId(uid);
        setIsLoggedIn(true);

        ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data as string) as ReactionMarker;
            if (
              data.reactionId &&
              data.latitude != null &&
              data.longitude != null &&
              typeof data.reactionType === "string" &&
              isValidReactionType(data.reactionType)
            ) {
              setReactions(prev => [...prev, data]);
            } else {
              console.warn("Dropping WebSocket reaction — missing fields or unknown reactionType:", data);
            }
          } catch {
            console.error("Failed to parse WebSocket message:", event.data);
          }
        };

        ws.onerror = (err) => console.error("WebSocket error:", err);
        ws.onclose = (evt) => console.warn(`WebSocket closed — code: ${evt.code}, reason: "${evt.reason}"`);
      } catch {
        // Not authenticated — no WebSocket
      }
    })();

    return () => {
      cancelled = true;
      ws?.close();
      wsRef.current = null;
    };
  }, []);

  const sendReaction = useCallback(
    (reactionType: ReactionType, message: string) => {
      if (wsRef.current?.readyState !== WebSocket.OPEN || !pendingLocation || !userId) return;
      wsRef.current.send(JSON.stringify({
        action: "sendReaction",
        reactionType,
        message,
        latitude: pendingLocation.lat,
        longitude: pendingLocation.lng,
        userId,
      }));
      setPendingLocation(null);
    },
    [pendingLocation, userId]
  );

  return (
    <Box sx={{
      display: "flex",
      justifyContent: "center",
      alignItems: "flex-start",
      height: "calc(100vh - 64px)",
      bgcolor: "#0f1b2d",
      p: 3,
    }}>
      <Box sx={{ flex: 1, height: "100%", minWidth: 0 }}>
        <Typography variant="h6" sx={{ color: "#fff", mb: 2 }}>
          Avalanche Risk Map
        </Typography>
        <Box sx={{ position: "relative", height: "calc(100% - 48px)", borderRadius: 3, overflow: "hidden" }}>
          <MapErrorBoundary>
            <MapComponent
              coords={submittedCoords}
              reactions={reactions}
              pendingLocation={pendingLocation}
              onLocationSelect={(lat, lng) => {
                if (isLoggedIn) setPendingLocation({ lat, lng });
              }}
            />
          </MapErrorBoundary>

          {/* Legend overlay */}
          <Box sx={{
            position: "absolute",
            bottom: 24,
            left: 16,
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

      <Sidebar
        onSubmit={setSubmittedCoords}
        sendReaction={sendReaction}
        pendingLocation={pendingLocation}
        isLoggedIn={isLoggedIn}
      />
    </Box>
  );
}
