import { Link } from "react-router-dom";
import { tw } from "../design/tokens";
import EmptyState from "../design/EmptyState";

export default function NotFound() {
  return (
    <EmptyState
      icon={
        <span className="text-4xl font-bold text-[#5e6673]">404</span>
      }
      title="Page not found"
      description="The page you're looking for doesn't exist or has been moved."
      action={
        <Link
          to="/"
          className="px-4 py-2 rounded-md bg-[#22d1ee]/10 border border-[#22d1ee]/30 text-sm font-medium text-[#22d1ee] hover:bg-[#22d1ee]/20 transition-colors"
        >
          Back to Overview
        </Link>
      }
    />
  );
}
