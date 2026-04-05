import { useEffect } from "react";
import { useMap } from "react-leaflet";
import * as L from "leaflet";
import "leaflet.markercluster";
import "leaflet.markercluster/dist/MarkerCluster.css";
import type { ReactionMarker, ReactionType } from "./MapComponent";

const REACTION_EMOJI: Record<ReactionType, string> = {
  icy: "❄️",
  powder: "⛷️",
  bluebird: "☀️",
  crowded: "👥",
  heavy_snow: "🌨️",
  foggy: "🌫️",
  sketchy: "⚠️",
  avalanche: "💀",
};

const REACTION_LABEL: Record<ReactionType, string> = {
  icy: "Icy",
  powder: "Powder",
  bluebird: "Bluebird",
  crowded: "Crowded",
  heavy_snow: "Heavy Snow",
  foggy: "Foggy",
  sketchy: "Sketchy",
  avalanche: "Avalanche",
};

function makeIcon(emoji: string): L.DivIcon {
  return L.divIcon({
    html: `<div style="font-size:22px;line-height:1;filter:drop-shadow(0 1px 3px rgba(0,0,0,0.7));">${emoji}</div>`,
    className: "",
    iconSize: [26, 26],
    iconAnchor: [13, 13],
  });
}

const reactionIcons: Record<ReactionType, L.DivIcon> = {
  icy: makeIcon("❄️"),
  powder: makeIcon("⛷️"),
  bluebird: makeIcon("☀️"),
  crowded: makeIcon("👥"),
  heavy_snow: makeIcon("🌨️"),
  foggy: makeIcon("🌫️"),
  sketchy: makeIcon("⚠️"),
  avalanche: makeIcon("💀"),
};

const fallbackIcon = L.divIcon({
  html: `<div style="font-size:22px;line-height:1;filter:drop-shadow(0 1px 3px rgba(0,0,0,0.7));">📌</div>`,
  className: "",
  iconSize: [26, 26],
  iconAnchor: [13, 13],
});

function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
}

function buildPopup(r: ReactionMarker): HTMLElement {
  const root = document.createElement("div");
  root.style.cssText = "min-width:160px;font-family:sans-serif";

  const emoji = document.createElement("div");
  emoji.style.cssText = "font-size:28px;line-height:1;margin-bottom:6px";
  emoji.textContent = REACTION_EMOJI[r.reactionType] ?? "📌";
  root.appendChild(emoji);

  const label = document.createElement("div");
  label.style.cssText =
    "font-weight:700;font-size:13px;color:#111;margin-bottom:6px";
  label.textContent = REACTION_LABEL[r.reactionType] ?? r.reactionType;
  root.appendChild(label);

  if (r.message) {
    const msg = document.createElement("div");
    msg.style.cssText =
      "font-size:13px;color:#333;line-height:1.5;margin-bottom:8px";
    msg.textContent = r.message;
    root.appendChild(msg);
  }

  const ts = document.createElement("div");
  ts.style.cssText = "font-size:11px;color:#888";
  ts.textContent = formatTimestamp(r.timestamp);
  root.appendChild(ts);

  return root;
}

interface Props {
  reactions: ReactionMarker[];
}

export default function ReactionClusterLayer({ reactions }: Props) {
  const map = useMap();

  useEffect(() => {
    const clusterGroup = L.markerClusterGroup({
      maxClusterRadius: 25,
      spiderfyOnMaxZoom: true,
      showCoverageOnHover: false,
      zoomToBoundsOnClick: true,
      iconCreateFunction: (cluster) => {
        const count = cluster.getChildCount();
        return L.divIcon({
          html: `<div style="
            background: rgba(21,101,192,0.88);
            color: #fff;
            border-radius: 50%;
            width: 36px; height: 36px;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 13px;
            border: 2px solid rgba(255,255,255,0.9);
            box-shadow: 0 2px 8px rgba(0,0,0,0.45);
          ">${count}</div>`,
          className: "",
          iconSize: [36, 36],
          iconAnchor: [18, 18],
        });
      },
    });

    reactions.forEach((r) => {
      const icon = reactionIcons[r.reactionType] ?? fallbackIcon;
      const marker = L.marker([r.latitude, r.longitude], { icon });
      marker.bindPopup(buildPopup(r));
      // Stop click from propagating to the map (prevents dropping a new pin)
      marker.on("click", (e) => L.DomEvent.stopPropagation(e));
      clusterGroup.addLayer(marker);
    });

    map.addLayer(clusterGroup);
    return () => {
      map.removeLayer(clusterGroup);
    };
  }, [reactions, map]);

  return null;
}
