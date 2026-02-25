import { cn } from "@/lib/utils";

interface WaveformVisualizerProps {
  isAnimating: boolean;
  barCount?: number;
  className?: string;
}

const WaveformVisualizer = ({ isAnimating, barCount = 40, className }: WaveformVisualizerProps) => {
  return (
    <div className={cn("flex items-center justify-center gap-[2px] h-16", className)}>
      {Array.from({ length: barCount }).map((_, i) => (
        <div
          key={i}
          className={cn(
            "w-1 rounded-full bg-primary/60 transition-all duration-300",
            isAnimating ? "animate-wave" : "h-2 bg-primary/20"
          )}
          style={{
            animationDelay: isAnimating ? `${i * 0.05}s` : undefined,
            height: isAnimating ? undefined : `${Math.random() * 20 + 4}px`,
          }}
        />
      ))}
    </div>
  );
};

export default WaveformVisualizer;
