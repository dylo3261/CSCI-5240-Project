import { useEffect, useMemo, useState } from "react";
import {
  MapContainer,
  TileLayer,
  ImageOverlay,
  Marker,
  useMap,
  useMapEvents,
} from "react-leaflet";
import * as L from "leaflet";
import "leaflet/dist/leaflet.css";
import ExplainabilityCard from "./ExplainabilityCard";
import ReactionClusterLayer from "./ReactionClusterLayer";

const LAT_STEP = 2 / 69;
const LON_STEP = 2 / 53;

// Exact Colorado state boundary corners (DMS → decimal degrees)
const COLORADO_BOUNDS: L.LatLngBoundsLiteral = [
  [36.9989, -109.0452], // SW — Four Corners Monument
  [41.0, -102.0467], // NE corner
];
const MIN_ZOOM = 6;

const getColor = (v: number) => `hsl(${(1 - v) * 120}, 90%, 45%)`;

// Canvas resolution for the grid overlay image.
// At 1400×800, each 2-mile cell is ~7×6 pixels — enough for seamless tiling.
const OVERLAY_WIDTH = 1400;
const OVERLAY_HEIGHT = 800;

export type ReactionType =
  | "icy"
  | "powder"
  | "bluebird"
  | "crowded"
  | "heavy_snow"
  | "foggy"
  | "sketchy"
  | "avalanche";

export interface ReactionMarker {
  reactionId: string;
  dataType: string;
  timestamp: string;
  reactionType: ReactionType;
  message?: string;
  latitude: number;
  longitude: number;
  userId?: string;
}

const pendingLocationIcon = L.divIcon({
  html: `<div style="font-size:24px;line-height:1;filter:drop-shadow(0 2px 4px rgba(0,0,0,0.8));">📍</div>`,
  className: "",
  iconSize: [24, 32],
  iconAnchor: [12, 32],
});

interface GridCell {
  lat: number;
  lon: number;
  value: number;
}

interface Props {
  coords: { lat: number; lng: number } | null;
  reactions: ReactionMarker[];
  pendingLocation: { lat: number; lng: number } | null;
  onLocationSelect: (lat: number, lng: number) => void;
}

function FlyTo({ coords }: { coords: { lat: number; lng: number } | null }) {
  const map = useMap();
  useEffect(() => {
    if (coords) {
      map.flyTo([coords.lat, coords.lng], 10, { duration: 1.5 });
    }
  }, [coords, map]);
  return null;
}

// Cell-key used for O(1) coverage lookup.
// Math.round(x / STEP) maps any point within half a step of a cell centre
// to the same integer, matching how cells are quantised in the data pipeline.
function cellKey(lat: number, lng: number): string {
  return `${Math.round(lat / LAT_STEP)},${Math.round(lng / LON_STEP)}`;
}

function LocationSelector({
  onLocationSelect,
  coverageSet,
}: {
  onLocationSelect: (lat: number, lng: number) => void;
  coverageSet: Set<string>;
}) {
  const map = useMap();

  useMapEvents({
    mousemove(e) {
      const covered =
        coverageSet.size === 0 // still loading — don't block
          ? true
          : coverageSet.has(cellKey(e.latlng.lat, e.latlng.lng));
      map.getContainer().style.cursor = covered ? "crosshair" : "not-allowed";
    },
    mouseout() {
      map.getContainer().style.cursor = "";
    },
    click(e) {
      if (coverageSet.size > 0 && coverageSet.has(cellKey(e.latlng.lat, e.latlng.lng))) {
        onLocationSelect(e.latlng.lat, e.latlng.lng);
      }
    },
  });
  return null;
}

export default function MapComponent({
  coords,
  reactions,
  pendingLocation,
  onLocationSelect,
}: Props) {
  const [cells, setCells] = useState<GridCell[]>([]);
  // Track which selected location has been dismissed so a new pin re-shows the card
  const [dismissedLocationKey, setDismissedLocationKey] = useState("");
  const [cardOpen, setCardOpen] = useState(false);
  const pendingLocationKey = pendingLocation
    ? `${pendingLocation.lat},${pendingLocation.lng}`
    : null;
  const cardDismissed =
    pendingLocationKey !== null && dismissedLocationKey === pendingLocationKey;

  // Reset card visibility whenever the selected location changes
  useEffect(() => {
    setCardOpen(false);
  }, [pendingLocationKey]);

  // Build a Set of quantised cell keys so LocationSelector can do O(1)
  // coverage checks on click and mousemove.
  const coverageSet = useMemo(() => {
    const set = new Set<string>();
    for (const cell of cells) {
      set.add(cellKey(cell.lat, cell.lon));
    }
    return set;
  }, [cells]);

  // Render grid cells to an offscreen canvas once cells are loaded.
  // Using floor/ceil for pixel bounds guarantees adjacent cells share edges
  // with no sub-pixel gap regardless of zoom level.
  const overlayUrl = useMemo(() => {
    if (cells.length === 0) return "";

    const [[south, west], [north, east]] = COLORADO_BOUNDS;
    const lonRange = east - west;
    const latRange = north - south;
    const W = OVERLAY_WIDTH;
    const H = OVERLAY_HEIGHT;

    const canvas = document.createElement("canvas");
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d")!;

    for (const cell of cells) {
      const x0 = Math.floor(((cell.lon - LON_STEP / 2) - west) / lonRange * W);
      const x1 = Math.ceil(((cell.lon + LON_STEP / 2) - west) / lonRange * W);
      const y0 = Math.floor((north - (cell.lat + LAT_STEP / 2)) / latRange * H);
      const y1 = Math.ceil((north - (cell.lat - LAT_STEP / 2)) / latRange * H);
      ctx.fillStyle = getColor(cell.value);
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    }

    return canvas.toDataURL();
  }, [cells]);

  useEffect(() => {
    fetch(
      "https://mera3wkzuj.execute-api.us-west-2.amazonaws.com/request-redirector",
    )
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<GridCell[]>;
      })
      .then((json) => {
        const data: GridCell[] = json
          .map((c) => ({
            lat: Number(c.lat),
            lon: Number(c.lon),
            value: Number(c.value),
          }))
          .filter(
            (c) =>
              !Number.isNaN(c.lat) &&
              !Number.isNaN(c.lon) &&
              !Number.isNaN(c.value),
          );
        setCells(data);
      })
      .catch((err) => console.error("Failed to load grid data:", err));
  }, []);

  return (
    // Wrapper needed so the card overlay can be positioned relative to the map
    <div style={{ position: "relative", height: "100%", width: "100%" }}>
      <MapContainer
        center={[39, -105.54]}
        zoom={7}
        minZoom={MIN_ZOOM}
        maxBounds={COLORADO_BOUNDS}
        maxBoundsViscosity={0.8}
        style={{ height: "100%", width: "100%" }}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
          attribution='© <a href="https://carto.com/">CARTO</a>'
        />
        <FlyTo coords={coords} />
        <LocationSelector onLocationSelect={onLocationSelect} coverageSet={coverageSet} />
        {overlayUrl && (
          <ImageOverlay
            url={overlayUrl}
            bounds={COLORADO_BOUNDS}
            opacity={0.55}
          />
        )}
        {pendingLocation && (
          <Marker
            position={[pendingLocation.lat, pendingLocation.lng]}
            icon={pendingLocationIcon}
          />
        )}

        <ReactionClusterLayer reactions={reactions} />
      </MapContainer>{" "}
      {/* Button shown when pin is placed but card not yet opened */}
      {pendingLocation && !cardDismissed && !cardOpen && (
        <button
          onClick={() => setCardOpen(true)}
          style={{
            position: "absolute",
            top: 16,
            right: 16,
            zIndex: 1000,
            display: "flex",
            alignItems: "center",
            gap: 6,
            padding: "7px 14px",
            background: "#0a1628",
            border: "1px solid rgba(255,255,255,0.15)",
            borderRadius: 20,
            color: "rgba(255,255,255,0.85)",
            fontSize: 12,
            fontWeight: 600,
            letterSpacing: 0.4,
            cursor: "pointer",
            boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
          }}
        >
          <span style={{ fontSize: 14 }}>ⓘ</span>
          Prediction Explanation
        </button>
      )}
      {/* Always mounted when pin is placed so fetch starts immediately */}
      {pendingLocation && !cardDismissed && (
        <ExplainabilityCard
          location={pendingLocation}
          open={cardOpen}
          onClose={() => {
            setCardOpen(false);
            if (pendingLocationKey !== null) {
              setDismissedLocationKey(pendingLocationKey);
            }
          }}
        />
      )}
    </div>
  );
}
