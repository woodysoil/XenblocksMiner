import { Component, type ErrorInfo, type ReactNode } from "react";
import { colors, tw, radius, space } from "../design/tokens";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  render() {
    if (!this.state.hasError) return this.props.children;

    const isDev = process.env.NODE_ENV !== "production";

    return (
      <div
        role="alert"
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "50vh",
          padding: space[8],
          textAlign: "center",
        }}
      >
        {/* Error icon */}
        <div
          style={{
            width: 56,
            height: 56,
            borderRadius: radius.full,
            backgroundColor: colors.danger.muted,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: space[5],
          }}
        >
          <svg
            width="28"
            height="28"
            viewBox="0 0 24 24"
            fill="none"
            stroke={colors.danger.DEFAULT}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
        </div>

        <h1
          style={{
            fontSize: "1.25rem",
            fontWeight: 600,
            color: colors.text.primary,
            margin: 0,
            marginBottom: space[2],
          }}
        >
          Something went wrong
        </h1>

        <p
          style={{
            fontSize: "0.875rem",
            color: colors.text.secondary,
            margin: 0,
            marginBottom: space[6],
            maxWidth: 420,
            lineHeight: 1.5,
          }}
        >
          An unexpected error occurred. Please try reloading the page.
        </p>

        {isDev && this.state.error && (
          <pre
            className={tw.card}
            style={{
              fontSize: "0.75rem",
              color: colors.danger.DEFAULT,
              padding: space[4],
              marginBottom: space[6],
              maxWidth: 560,
              overflow: "auto",
              textAlign: "left",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {this.state.error.message}
            {"\n\n"}
            {this.state.error.stack}
          </pre>
        )}

        <button
          className={tw.btnPrimary}
          onClick={() => window.location.reload()}
        >
          Try Again
        </button>
      </div>
    );
  }
}
