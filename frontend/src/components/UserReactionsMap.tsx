import { useEffect } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import * as L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { ReactionMarker } from "./MapComponent";
import ReactionClusterLayer from "./ReactionClusterLayer";

function FitBounds({ reactions }: { reactions: ReactionMarker[] }) {
  const map = useMap();
  useEffect(() => {
    if (reactions.length === 0) return;
    const bounds = L.latLngBounds(reactions.map(r => [r.latitude, r.longitude]));
    map.fitBounds(bounds, { padding: [48, 48], maxZoom: 12 });
  }, [reactions, map]);
  return null;
}

interface Props {
  reactions: ReactionMarker[];
}

export default function UserReactionsMap({ reactions }: Props) {
  return (
    <MapContainer
      center={[39, -105.54]}
      zoom={7}
      style={{ height: "100%", width: "100%" }}
    >
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
        attribution='© <a href="https://carto.com/">CARTO</a>'
      />
      <FitBounds reactions={reactions} />
      <ReactionClusterLayer reactions={reactions} />
    </MapContainer>
  );
}
