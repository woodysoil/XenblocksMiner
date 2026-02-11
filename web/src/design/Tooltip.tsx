import { type ReactNode } from 'react';
import { colors, radius } from './tokens';

interface TooltipProps {
  content: string;
  children: ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
}

const ARROW = 5;

const positionStyles: Record<string, React.CSSProperties> = {
  top: { bottom: '100%', left: '50%', transform: 'translateX(-50%)', marginBottom: ARROW + 2 },
  bottom: { top: '100%', left: '50%', transform: 'translateX(-50%)', marginTop: ARROW + 2 },
  left: { right: '100%', top: '50%', transform: 'translateY(-50%)', marginRight: ARROW + 2 },
  right: { left: '100%', top: '50%', transform: 'translateY(-50%)', marginLeft: ARROW + 2 },
};

const arrowStyles: Record<string, React.CSSProperties> = {
  top: {
    bottom: -ARROW,
    left: '50%',
    transform: 'translateX(-50%)',
    borderLeft: `${ARROW}px solid transparent`,
    borderRight: `${ARROW}px solid transparent`,
    borderTop: `${ARROW}px solid ${colors.bg.surface1}`,
  },
  bottom: {
    top: -ARROW,
    left: '50%',
    transform: 'translateX(-50%)',
    borderLeft: `${ARROW}px solid transparent`,
    borderRight: `${ARROW}px solid transparent`,
    borderBottom: `${ARROW}px solid ${colors.bg.surface1}`,
  },
  left: {
    right: -ARROW,
    top: '50%',
    transform: 'translateY(-50%)',
    borderTop: `${ARROW}px solid transparent`,
    borderBottom: `${ARROW}px solid transparent`,
    borderLeft: `${ARROW}px solid ${colors.bg.surface1}`,
  },
  right: {
    left: -ARROW,
    top: '50%',
    transform: 'translateY(-50%)',
    borderTop: `${ARROW}px solid transparent`,
    borderBottom: `${ARROW}px solid transparent`,
    borderRight: `${ARROW}px solid ${colors.bg.surface1}`,
  },
};

const tooltipCss = `
.xb-tooltip .xb-tip {
  opacity: 0;
  pointer-events: none;
  transition: opacity 150ms ease;
  transition-delay: 0ms;
}
.xb-tooltip:hover .xb-tip {
  opacity: 1;
  pointer-events: auto;
  transition-delay: 200ms;
}
`;

export default function Tooltip({ content, children, position = 'top' }: TooltipProps) {
  return (
    <>
      <style>{tooltipCss}</style>
      <span className="xb-tooltip relative inline-flex">
        {children}
        <span
          className="xb-tip absolute z-50 whitespace-normal"
          style={{
            ...positionStyles[position],
            maxWidth: 200,
            padding: '6px 10px',
            fontSize: 12,
            lineHeight: 1.4,
            color: colors.text.primary,
            backgroundColor: colors.bg.surface1,
            borderRadius: radius.md,
            wordWrap: 'break-word',
          }}
        >
          {content}
          <span className="absolute" style={{ width: 0, height: 0, ...arrowStyles[position] }} />
        </span>
      </span>
    </>
  );
}
