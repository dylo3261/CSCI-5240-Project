import { useEffect, useState } from "react";
import {
  MapContainer,
  TileLayer,
  Rectangle,
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

const canvasRenderer = L.canvas({ padding: 0.5 });

const getColor = (v: number) => `hsl(${(1 - v) * 240}, 90%, 50%)`;

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
  }, [coords]);
  return null;
}

function LocationSelector({
  onLocationSelect,
}: {
  onLocationSelect: (lat: number, lng: number) => void;
}) {
  useMapEvents({
    click(e) {
      onLocationSelect(e.latlng.lat, e.latlng.lng);
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
  const pendingLocationKey = pendingLocation
    ? `${pendingLocation.lat},${pendingLocation.lng}`
    : null;
  const cardDismissed =
    pendingLocationKey !== null && dismissedLocationKey === pendingLocationKey;

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
        style={{ height: "100%", width: "100%" }}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
          attribution='© <a href="https://carto.com/">CARTO</a>'
        />
        <FlyTo coords={coords} />
        <LocationSelector onLocationSelect={onLocationSelect} />
        {cells.map((cell, i) => (
          <Rectangle
            key={i}
            bounds={[
              [cell.lat - LAT_STEP / 2, cell.lon - LON_STEP / 2],
              [cell.lat + LAT_STEP / 2, cell.lon + LON_STEP / 2],
            ]}
            renderer={canvasRenderer}
            pathOptions={{
              fillColor: getColor(cell.value),
              fillOpacity: 0.35,
              stroke: false,
            }}
          />
        ))}
        {pendingLocation && (
          <Marker
            position={[pendingLocation.lat, pendingLocation.lng]}
            icon={pendingLocationIcon}
          />
        )}

        <ReactionClusterLayer reactions={reactions} />
      </MapContainer>{" "}
      {!cardDismissed && (
        <ExplainabilityCard
          location={pendingLocation}
          onClose={() => {
            if (pendingLocationKey !== null) {
              setDismissedLocationKey(pendingLocationKey);
            }
          }}
        />
      )}
    </div>
  );
}
