import { useState, useEffect, useRef, useCallback, Component, type ReactNode, type ErrorInfo } from "react";
import { Box, Typography } from "@mui/material";
import { fetchAuthSession, getCurrentUser } from "aws-amplify/auth";
import MapComponent, { type ReactionMarker, type ReactionType } from "../components/MapComponent";
import Sidebar from "../components/Sidebar";

const VALID_REACTION_TYPES = new Set<string>([
  "icy", "powder", "bluebird", "crowded", "heavy_snow", "foggy", "sketchy", "avalanche",
]);

function isValidReactionType(value: string): value is ReactionType {
  return VALID_REACTION_TYPES.has(value);
}

function useIsMobile() {
  const [isMobile, setIsMobile] = useState(() => window.innerWidth < 768);
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 767px)");
    const handler = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);
  return isMobile;
}

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
        <Box sx={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", bgcolor: "#0D0A1A" }}>
          <Typography color="error" variant="body2">
            Map failed to render a pin. Reload the map to retry.
          </Typography>
        </Box>
      );
    }
    return this.props.children;
  }
}

const WS_URL = "wss://j9jkzycsge.execute-api.us-west-2.amazonaws.com/prod";
const FETCH_REACTIONS_URL = "https://wpvtd43yyi.execute-api.us-west-2.amazonaws.com/reactions";
const getColor = (v: number) => `hsl(${(1 - v) * 120}, 90%, 45%)`;

export default function Map() {
  const [submittedCoords, setSubmittedCoords] = useState<{ lat: number; lng: number } | null>(null);
  const [reactions, setReactions] = useState<ReactionMarker[]>([]);
  const [pendingLocation, setPendingLocation] = useState<{ lat: number; lng: number } | null>(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userId, setUserId] = useState("");
  const [sheetOpen, setSheetOpen] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const sheetRef = useRef<HTMLDivElement>(null);
  const dragStartY = useRef(0);
  const dragOffset = useRef(0);
  const isMobile = useIsMobile();

  function onDragHandleTouchStart(e: React.TouchEvent) {
    dragStartY.current = e.touches[0].clientY;
    dragOffset.current = 0;
    if (sheetRef.current) sheetRef.current.style.transition = "none";
  }

  function onDragHandleTouchMove(e: React.TouchEvent) {
    const delta = e.touches[0].clientY - dragStartY.current;
    if (delta < 0) return;
    dragOffset.current = delta;
    if (sheetRef.current) sheetRef.current.style.transform = `translateY(${delta}px)`;
  }

  function onDragHandleTouchEnd() {
    if (sheetRef.current) sheetRef.current.style.transition = "";
    if (dragOffset.current > 80) {
      if (sheetRef.current) sheetRef.current.style.transform = "";
      setSheetOpen(false);
    } else {
      if (sheetRef.current) sheetRef.current.style.transform = "";
    }
    dragOffset.current = 0;
  }

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
        if (!cancelled && session.tokens) {
          const { userId: uid } = await getCurrentUser();
          if (!cancelled) { setUserId(uid); setIsLoggedIn(true); }
        }
      } catch { /* anonymous */ }

      if (cancelled) return;
      ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data as string) as ReactionMarker;
          if (
            data.reactionId && data.latitude != null && data.longitude != null &&
            typeof data.reactionType === "string" && isValidReactionType(data.reactionType)
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
    })();

    return () => { cancelled = true; ws?.close(); wsRef.current = null; };
  }, []);

  const sendReaction = useCallback(
    (reactionType: ReactionType, message: string) => {
      if (wsRef.current?.readyState !== WebSocket.OPEN || !pendingLocation) return;
      const payload: Record<string, unknown> = {
        action: "sendReaction", reactionType, message,
        latitude: pendingLocation.lat, longitude: pendingLocation.lng,
      };
      if (userId) payload.userId = userId;
      wsRef.current.send(JSON.stringify(payload));
      setPendingLocation(null);
      setSheetOpen(false);
    },
    [pendingLocation, userId]
  );

  const handleLocationSelect = useCallback((lat: number, lng: number) => {
    setPendingLocation({ lat, lng });
    if (isMobile) setSheetOpen(true);
  }, [isMobile]);

  const FOOTER_H = 36;
  const NAVBAR_H = 48;

  return (
    <Box sx={{
      display: "flex",
      flexDirection: "column",
      height: `calc(100vh - ${NAVBAR_H}px - env(safe-area-inset-bottom, 16px))`,
      bgcolor: "#0D0A1A",
    }}>

      {/* ── Map + Sidebar ── */}
      <Box sx={{
        flex: 1,
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
        overflow: "hidden",
        minHeight: 0,
        p: 1.5,
        gap: 1.5,
        position: "relative",
      }}>

        {/* ── Map ── */}
        <Box sx={{ flex: 1, height: "100%", minWidth: 0, position: "relative" }}>
          <Box sx={{ position: "relative", height: "100%", borderRadius: 3, overflow: "hidden" }}>
            <MapErrorBoundary>
              <MapComponent
                coords={submittedCoords}
                reactions={reactions}
                pendingLocation={pendingLocation}
                onLocationSelect={handleLocationSelect}
              />
            </MapErrorBoundary>

            {/* Legend overlay */}
            <Box sx={{
              position: "absolute",
              bottom: isMobile ? 60 : 20,
              left: 14,
              zIndex: 1000,
              bgcolor: "rgba(13, 10, 26, 0.88)",
              backdropFilter: "blur(8px)",
              border: "1px solid rgba(255,45,120,0.2)",
              borderRadius: 2,
              p: 1.5,
              minWidth: 150,
            }}>
              <Typography variant="caption" color="rgba(255,255,255,0.4)"
                display="block" mb={1} letterSpacing={0.5} textTransform="uppercase" fontSize={10}>
                Slab Risk
              </Typography>
              <Box sx={{
                height: 10, borderRadius: 1, mb: 0.75,
                background: `linear-gradient(to right, ${Array.from({ length: 10 }, (_, i) => getColor(i / 9)).join(", ")})`,
              }} />
              <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                <Typography variant="caption" color="rgba(255,255,255,0.3)" fontSize={10}>Low</Typography>
                <Typography variant="caption" color="rgba(255,255,255,0.3)" fontSize={10}>High</Typography>
              </Box>
            </Box>

            {/* Mobile: subtle persistent handle */}
            {isMobile && (
              <Box
                onClick={() => setSheetOpen(true)}
                sx={{
                  position: "absolute", bottom: 16, left: "50%", transform: "translateX(-50%)",
                  zIndex: 1000, bgcolor: "rgba(13,10,26,0.75)", backdropFilter: "blur(8px)",
                  border: "1px solid rgba(255,45,120,0.25)", borderRadius: 99,
                  px: 2, py: 0.75, cursor: "pointer",
                }}
              >
                <Typography variant="caption" color="rgba(240,248,255,0.6)" fontSize={12}>
                  {pendingLocation ? "📍 Pin dropped — tap to share" : "Tap map · share conditions"}
                </Typography>
              </Box>
            )}
          </Box>
        </Box>

        {/* ── Desktop: Sidebar ── */}
        {!isMobile && (
          <Box sx={{ height: "100%", flexShrink: 0 }}>
            <Sidebar
              onSubmit={setSubmittedCoords}
              sendReaction={sendReaction}
              pendingLocation={pendingLocation}
              isLoggedIn={isLoggedIn}
            />
          </Box>
        )}

        {/* ── Mobile: Bottom Sheet ── */}
        {isMobile && (
          <>
            {/* Backdrop */}
            {sheetOpen && (
              <Box
                onClick={() => setSheetOpen(false)}
                sx={{
                  position: "absolute", inset: 0, zIndex: 1100,
                  bgcolor: "rgba(0,0,0,0.4)", backdropFilter: "blur(2px)",
                }}
              />
            )}

            {/* Sheet */}
            <Box ref={sheetRef} sx={{
              position: "absolute", bottom: 0, left: 0, right: 0,
              zIndex: 1200,
              maxHeight: "80vh",
              bgcolor: "#150E2A",
              borderTop: "1px solid rgba(255,45,120,0.2)",
              borderRadius: "20px 20px 0 0",
              boxShadow: "0 -8px 40px rgba(0,0,0,0.6)",
              transform: sheetOpen ? "translateY(0)" : "translateY(100%)",
              transition: "transform 0.35s cubic-bezier(0.4, 0, 0.2, 1)",
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}>
              {/* Drag handle — touch target for swipe-to-close */}
              <Box
                onTouchStart={onDragHandleTouchStart}
                onTouchMove={onDragHandleTouchMove}
                onTouchEnd={onDragHandleTouchEnd}
                sx={{ display: "flex", justifyContent: "center", pt: 1.5, pb: 0.5, flexShrink: 0, touchAction: "none" }}
              >
                <Box sx={{ width: 40, height: 4, borderRadius: 2, bgcolor: "rgba(240,248,255,0.2)" }} />
              </Box>

              {/* Scrollable Sidebar content */}
              <Box sx={{ overflowY: "auto", flex: 1 }}>
                <Sidebar
                  onSubmit={setSubmittedCoords}
                  sendReaction={sendReaction}
                  pendingLocation={pendingLocation}
                  isLoggedIn={isLoggedIn}
                />
              </Box>
            </Box>
          </>
        )}
      </Box>

      {/* ── Footer (desktop only) ── */}
      {!isMobile && (
        <Box sx={{
          height: FOOTER_H,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 3,
          borderTop: "1px solid rgba(240,248,255,0.06)",
          px: 3,
          flexShrink: 0,
        }}>
          <Typography variant="caption" color="rgba(240,248,255,0.25)" fontSize={10} letterSpacing="0.06em">
            SLAB LAB · Colorado Backcountry Intelligence
          </Typography>
          <Typography
            component="a"
            href="https://avalanche.state.co.us"
            target="_blank"
            rel="noopener noreferrer"
            variant="caption"
            sx={{ color: "rgba(240,248,255,0.25)", fontSize: 10, letterSpacing: "0.06em", textDecoration: "none", "&:hover": { color: "#00E5CC" } }}
          >
            CAIC Data
          </Typography>
          <Typography variant="caption" color="rgba(240,248,255,0.25)" fontSize={10} letterSpacing="0.06em">
            Not a substitute for official forecasts
          </Typography>
        </Box>
      )}
    </Box>
  );
}