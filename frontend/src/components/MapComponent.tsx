import { useEffect, useState } from "react";
import { MapContainer, TileLayer, Rectangle, useMap } from "react-leaflet";
import L from "leaflet";
import Papa from "papaparse";
import "leaflet/dist/leaflet.css";

const LAT_STEP = 2 / 69;
const LON_STEP = 2 / 53;

const canvasRenderer = L.canvas({ padding: 0.5 });

// value 0–1 → color (you can swap this out later)
const getColor = (v: number) => `hsl(${(1 - v) * 240}, 90%, 50%)`;

interface GridCell {
  lat: number;
  lon: number;
  value: number;
}

interface Props {
  coords: { lat: number; lng: number } | null;
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

export default function MapComponent({ coords }: Props) {
  const [cells, setCells] = useState<GridCell[]>([]);

  useEffect(() => {
    // Swap this for JSON fetch when it's ready
    fetch("/colorado_mountains_2mi.csv")
      .then(r => r.text())
      .then(csv => {
        const { data } = Papa.parse(csv, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
        });
        setCells(data as GridCell[]);
      })
      .catch(err => console.error("Failed to load grid data:", err));
  }, []);

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
      <FlyTo coords={coords} />
      {cells.map((cell, i) => (
        <Rectangle
          key={i}
          bounds={[
            [cell.lat, cell.lon],
            [cell.lat + LAT_STEP, cell.lon + LON_STEP],
          ]}
          renderer={canvasRenderer}
          pathOptions={{
            fillColor: getColor(cell.value),
            fillOpacity: 0.35,
            stroke: false,
          }}
        />
      ))}
    </MapContainer>
  );
}