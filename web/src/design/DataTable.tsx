import { type ReactNode, useState } from 'react';
import { tw } from './tokens';

export interface Column<T = Record<string, unknown>> {
  key: string;
  label: string;
  sortable?: boolean;
  render?: (value: unknown, row: T) => ReactNode;
}

interface DataTableProps<T = Record<string, unknown>> {
  columns: Column<T>[];
  data: T[];
  onSort?: (key: string, direction: 'asc' | 'desc') => void;
}

export default function DataTable<T extends Record<string, unknown>>({
  columns,
  data,
  onSort,
}: DataTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  const handleSort = (key: string) => {
    const next = sortKey === key && sortDir === 'asc' ? 'desc' : 'asc';
    setSortKey(key);
    setSortDir(next);
    onSort?.(key, next);
  };

  return (
    <div className={`${tw.card} overflow-hidden`}>
      <table className="w-full">
        <thead>
          <tr className={tw.surface2}>
            {columns.map((col) => (
              <th
                key={col.key}
                className={`${tw.tableHeader} px-4 py-3 text-left ${col.sortable ? 'cursor-pointer select-none' : ''}`}
                onClick={col.sortable ? () => handleSort(col.key) : undefined}
              >
                <span className="inline-flex items-center gap-1">
                  {col.label}
                  {col.sortable && sortKey === col.key && (
                    <span className="text-[10px]">{sortDir === 'asc' ? '▲' : '▼'}</span>
                  )}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} className={tw.tableRow}>
              {columns.map((col) => (
                <td key={col.key} className={tw.tableCell}>
                  {col.render ? col.render(row[col.key], row) : String(row[col.key] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
