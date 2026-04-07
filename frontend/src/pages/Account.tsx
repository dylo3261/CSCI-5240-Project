import { useEffect, useState, type Dispatch, type SetStateAction } from "react";
import {
  Box,
  Typography,
  Avatar,
  Divider,
  Button,
  Paper,
  Tabs,
  Tab,
  CircularProgress,
  Slider,
} from "@mui/material";
import { Authenticator } from "@aws-amplify/ui-react";
import { fetchUserAttributes, type FetchUserAttributesOutput, type AuthUser } from "aws-amplify/auth";
import type { UseAuthenticator } from "@aws-amplify/ui-react-core";
import type { ReactionMarker, ReactionType } from "../components/MapComponent";
import UserReactionsMap from "../components/UserReactionsMap";
import "./Account.css";

// Same base API Gateway as FETCH_REACTIONS_URL in Map.tsx
const FETCH_USER_REACTIONS_URL = "https://wpvtd43yyi.execute-api.us-west-2.amazonaws.com/user-reactions";

const VALID_REACTION_TYPES = new Set<string>([
  "icy", "powder", "bluebird", "crowded", "heavy_snow", "foggy", "sketchy", "avalanche",
]);

function isValidReactionType(value: string): value is ReactionType {
  return VALID_REACTION_TYPES.has(value);
}

const ONE_WEEK_MS = 7 * 24 * 60 * 60 * 1000;

function formatSliderLabel(ms: number): string {
  return new Date(ms).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

// Custom header injected into the Amplify Authenticator
const components = {
  Header() {
    return (
      <Box className="auth-custom-header">
        <Typography variant="h5" component="h1" className="auth-title">
          SLAB LAB
        </Typography>
        <Typography variant="caption" component="p" className="auth-subtitle">
          Your Mountain Dashboard
        </Typography>
      </Box>
    );
  },
};

// Strict typing for the sub-component props to satisfy linters
interface AccountDetailsProps {
  user?: AuthUser;
  signOut?: UseAuthenticator["signOut"];
  attributes: FetchUserAttributesOutput | null;
  setAttributes: Dispatch<SetStateAction<FetchUserAttributesOutput | null>>;
}

export default function Account() {
  const [attributes, setAttributes] = useState<FetchUserAttributesOutput | null>(null);

  return (
    <Box className="account-page-wrapper">
      <Authenticator
        loginMechanisms={["email"]}
        components={components}
        signUpAttributes={["name"]}
        formFields={{
          signUp: {
            name: {
              label: "Full Name",
              placeholder: "Enter your full name",
              order: 1,
              isRequired: true,
            },
          },
        }}
      >
        {({ signOut, user }) => (
          <AccountDetails
            user={user}
            signOut={signOut}
            attributes={attributes}
            setAttributes={setAttributes}
          />
        )}
      </Authenticator>
    </Box>
  );
}

// Sub-component handling the authenticated state
function AccountDetails({ user, signOut, attributes, setAttributes }: AccountDetailsProps) {
  const [tab, setTab] = useState(0);

  useEffect(() => {
    async function loadAttributes() {
      try {
        const attrs = await fetchUserAttributes();
        setAttributes(attrs);
      } catch (err) {
        console.error("Error fetching user attributes:", err);
      }
    }

    if (!attributes) {
      loadAttributes();
    }
  }, [attributes, setAttributes]);

  const initial = attributes?.name ? attributes.name.charAt(0).toUpperCase() : "U";

  return (
    <Paper
      elevation={3}
      className="profile-card"
      style={{ maxWidth: tab === 1 ? 900 : 450 }}
    >
      <Tabs
        value={tab}
        onChange={(_, v: number) => setTab(v)}
        sx={{ mb: 2, borderBottom: "1px solid #e2e8f0" }}
      >
        <Tab label="Profile" />
        <Tab label="My Reactions" />
      </Tabs>

      {tab === 0 && (
        <>
          <Typography variant="h6" component="h2" className="profile-heading">
            Your Profile
          </Typography>

          <Box className="profile-info-container">
            <Avatar className="profile-avatar">{initial}</Avatar>
            <Box className="profile-details">
              <Typography variant="subtitle1" component="h3" className="profile-name">
                {attributes?.name || "Loading..."}
              </Typography>
              <Typography variant="body2" className="profile-email">
                {attributes?.email || user?.signInDetails?.loginId}
              </Typography>
            </Box>
          </Box>

          <Divider className="profile-divider" />

          <Box className="profile-meta">
            <Typography variant="caption" component="p" className="meta-label">
              System User ID
            </Typography>
            <Typography variant="body2" component="p" className="meta-value">
              {user?.userId}
            </Typography>
          </Box>

          <Button
            variant="outlined"
            fullWidth
            className="btn-signout"
            onClick={signOut}
          >
            Secure Sign Out
          </Button>
        </>
      )}

      {tab === 1 && (
        <MyReactionsTab userId={user?.userId} />
      )}
    </Paper>
  );
}

function MyReactionsTab({ userId }: { userId: string | undefined }) {
  const [reactions, setReactions] = useState<ReactionMarker[]>([]);
  const [loading, setLoading] = useState(Boolean(userId));
  const [range, setRange] = useState<[number, number]>([0, 0]);

  useEffect(() => {
    if (!userId) {
      return;
    }

    setLoading(true);

    fetch(`${FETCH_USER_REACTIONS_URL}?userId=${encodeURIComponent(userId)}`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<ReactionMarker[]>;
      })
      .then(data => {
        const valid = data.filter(r => isValidReactionType(r.reactionType));
        setReactions(valid);
        if (valid.length > 0) {
          const first = new Date(valid[0].timestamp).getTime();
          const last = new Date(valid[valid.length - 1].timestamp).getTime();
          setRange([first, last]);
        }
      })
      .catch(err => console.error("Failed to load user reactions:", err))
      .finally(() => setLoading(false));
  }, [userId]);

  if (!userId) {
    return <Typography color="text.secondary">User ID not available.</Typography>;
  }

  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", py: 6 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (reactions.length === 0) {
    return (
      <Typography color="text.secondary" sx={{ py: 2 }}>
        You haven't posted any reactions yet. Head to the map to drop your first one!
      </Typography>
    );
  }

  const firstTs = new Date(reactions[0].timestamp).getTime();
  const lastTs = new Date(reactions[reactions.length - 1].timestamp).getTime();
  const showSlider = lastTs - firstTs > ONE_WEEK_MS;

  const filtered = reactions.filter(r => {
    const ts = new Date(r.timestamp).getTime();
    return ts >= range[0] && ts <= range[1];
  });

  return (
    <Box>
      {showSlider && (
        <Box sx={{ px: 1, mb: 3 }}>
          <Typography variant="caption" sx={{ color: "#64748b", display: "block", mb: 1 }}>
            Filter by date range
          </Typography>
          <Slider
            value={range}
            min={firstTs}
            max={lastTs}
            onChange={(_, v) => setRange(v as [number, number])}
            valueLabelDisplay="auto"
            valueLabelFormat={formatSliderLabel}
            disableSwap
            sx={{
              color: "#FF2D78",
              "& .MuiSlider-valueLabel": {
                fontSize: 11,
                bgcolor: "#0f172a",
              },
            }}
          />
          <Box sx={{ display: "flex", justifyContent: "space-between", mt: 0.5 }}>
            <Typography variant="caption" color="text.secondary">
              {formatSliderLabel(firstTs)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {formatSliderLabel(lastTs)}
            </Typography>
          </Box>
        </Box>
      )}

      <Typography variant="caption" sx={{ color: "#64748b", display: "block", mb: 1 }}>
        {filtered.length} reaction{filtered.length !== 1 ? "s" : ""} shown
      </Typography>

      <Box sx={{ height: 450, borderRadius: 2, overflow: "hidden", border: "1px solid #e2e8f0" }}>
        <UserReactionsMap reactions={filtered} />
      </Box>
    </Box>
  );
}
