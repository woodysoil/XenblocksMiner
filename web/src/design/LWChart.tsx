import { useEffect, useRef } from "react";
import {
  createChart,
  AreaSeries,
  ColorType,
  CrosshairMode,
  type IChartApi,
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
}

export default function LWChart({ data, height = 260, formatValue }: LWChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

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
      },
      localization: {
        ...(formatValue ? { priceFormatter: formatValue } : {}),
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
      ...(formatValue ? { priceFormat: { type: "custom" as const, formatter: formatValue } } : {}),
    });

    series.setData(data);
    chart.timeScale().fitContent();
    chartRef.current = chart;

    const ro = new ResizeObserver(([entry]) => {
      chart.applyOptions({ width: entry.contentRect.width });
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [data, height, formatValue]);

  return <div ref={containerRef} />;
}
