import { useEffect, useRef, useCallback } from "react";
import {
  createChart,
  AreaSeries,
  ColorType,
  CrosshairMode,
  type IChartApi,
  type ISeriesApi,
  type SeriesType,
  type UTCTimestamp,
} from "lightweight-charts";
import { colors } from "./tokens";

export interface LWDataPoint {
  time: UTCTimestamp;
  value: number;
}

interface LWChartProps {
  data: LWDataPoint[];
  height?: number;
  formatValue?: (v: number) => string;
  /** Default visible window in seconds. 0 = show all. */
  visibleWindow?: number;
}

export default function LWChart({ data, height = 260, formatValue, visibleWindow = 0 }: LWChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<SeriesType> | null>(null);

  const stableFormatter = useRef(formatValue);
  stableFormatter.current = formatValue;

  const ensureChart = useCallback(() => {
    if (chartRef.current) return chartRef.current;
    const el = containerRef.current;
    if (!el || el.clientWidth === 0) return null;

    const fmt = stableFormatter.current;
    const chart = createChart(el, {
      width: el.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: colors.text.secondary,
        fontFamily: "'Inter', system-ui, sans-serif",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: colors.bg.surface3 },
        horzLines: { color: colors.bg.surface3 },
      },
      timeScale: {
        borderColor: colors.border.default,
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: colors.border.default,
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "rgba(34,209,238,0.2)", labelBackgroundColor: "#1a2029" },
        horzLine: { color: "rgba(34,209,238,0.2)", labelBackgroundColor: "#1a2029" },
      },
      localization: {
        locale: "en-US",
        ...(fmt ? { priceFormatter: fmt } : {}),
      },
    });

    const series = chart.addSeries(AreaSeries, {
      lineColor: colors.accent.DEFAULT,
      topColor: colors.accent.glow,
      bottomColor: "rgba(34,209,238,0)",
      lineWidth: 2,
      crosshairMarkerRadius: 4,
      crosshairMarkerBorderWidth: 1,
      crosshairMarkerBorderColor: colors.accent.DEFAULT,
      crosshairMarkerBackgroundColor: colors.bg.surface1,
      ...(fmt ? { priceFormat: { type: "custom" as const, formatter: fmt } } : {}),
    });

    chartRef.current = chart;
    seriesRef.current = series;

    const ro = new ResizeObserver(([entry]) => {
      if (chartRef.current) {
        chartRef.current.applyOptions({ width: entry.contentRect.width });
      }
    });
    ro.observe(el);
    (chart as any).__ro = ro;

    return chart;
  }, [height]);

  useEffect(() => {
    return () => {
      if (chartRef.current) {
        const ro = (chartRef.current as any).__ro as ResizeObserver | undefined;
        ro?.disconnect();
        chartRef.current.remove();
        chartRef.current = null;
        seriesRef.current = null;
      }
    };
  }, []);

  // Data sync
  useEffect(() => {
    if (!data.length) return;
    const chart = ensureChart();
    if (!chart || !seriesRef.current) return;

    seriesRef.current.setData(data);
    applyVisibleWindow(chart, data, visibleWindow);
  }, [data, ensureChart, visibleWindow]);

  return <div ref={containerRef} />;
}

function applyVisibleWindow(chart: IChartApi, data: LWDataPoint[], windowSec: number) {
  if (!data.length) return;
  const last = data[data.length - 1].time as number;
  const first = data[0].time as number;
  const span = last - first;

  if (windowSec <= 0 || span <= windowSec) {
    chart.timeScale().fitContent();
  } else {
    chart.timeScale().setVisibleRange({
      from: (last - windowSec) as UTCTimestamp,
      to: last as UTCTimestamp,
    });
  }
}
