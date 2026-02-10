import React from "react";
import { tw } from "../design/tokens";

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

export default function Pagination({
  currentPage,
  totalPages,
  onPageChange,
}: PaginationProps) {
  if (totalPages <= 1) return null;

  const getPageNumbers = () => {
    const pages: (number | string)[] = [];
    const maxVisible = 7;

    if (totalPages <= maxVisible) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      pages.push(1);
      if (currentPage > 3) {
        pages.push("...");
      }

      const start = Math.max(2, currentPage - 1);
      const end = Math.min(totalPages - 1, currentPage + 1);

      for (let i = start; i <= end; i++) {
        pages.push(i);
      }

      if (currentPage < totalPages - 2) {
        pages.push("...");
      }
      pages.push(totalPages);
    }
    return pages;
  };

  return (
    <div className="flex items-center justify-center gap-2 mt-6 select-none">
      <button
        onClick={() => onPageChange(Math.max(1, currentPage - 1))}
        disabled={currentPage === 1}
        className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
          currentPage === 1
            ? "text-[#5e6673] cursor-not-allowed"
            : "text-[#848e9c] hover:text-[#eaecef] hover:bg-[#1f2835]"
        }`}
      >
        Prev
      </button>

      {getPageNumbers().map((page, i) => (
        <React.Fragment key={i}>
          {typeof page === "number" ? (
            <button
              onClick={() => onPageChange(page)}
              className={`min-w-[32px] h-8 flex items-center justify-center rounded-md text-xs font-medium transition-colors ${
                currentPage === page
                  ? "bg-[#22d1ee]/10 text-[#22d1ee] border border-[#22d1ee]/20"
                  : "text-[#848e9c] hover:text-[#eaecef] hover:bg-[#1f2835]"
              }`}
            >
              {page}
            </button>
          ) : (
            <span className="text-[#5e6673] px-1">...</span>
          )}
        </React.Fragment>
      ))}

      <button
        onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
        disabled={currentPage === totalPages}
        className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
          currentPage === totalPages
            ? "text-[#5e6673] cursor-not-allowed"
            : "text-[#848e9c] hover:text-[#eaecef] hover:bg-[#1f2835]"
        }`}
      >
        Next
      </button>

      <div className="ml-4 text-xs text-[#5e6673]">
        Page <span className="text-[#eaecef]">{currentPage}</span> of{" "}
        <span className="text-[#eaecef]">{totalPages}</span>
      </div>
    </div>
  );
}
