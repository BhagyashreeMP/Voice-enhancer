import { Zap, Shield, Gauge, Waves } from "lucide-react";

const features = [
  {
    icon: Zap,
    title: "AI-Powered Enhancement",
    description: "Deep learning algorithms analyze and enhance every frequency in your audio for crystal-clear results.",
  },
  {
    icon: Shield,
    title: "Noise Reduction",
    description: "Eliminate background noise, hum, and hiss while preserving the natural quality of your audio.",
  },
  {
    icon: Gauge,
    title: "Instant Processing",
    description: "Get studio-quality results in seconds, not hours. Our cloud engine handles the heavy lifting.",
  },
  {
    icon: Waves,
    title: "Format Support",
    description: "Works with MP3, WAV, FLAC, AAC, and more. Upload any audio format and download enhanced output.",
  },
];

const FeaturesSection = () => {
  return (
    <section className="py-20 px-4">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-14">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-3">
            Why <span className="text-gradient">Texutra</span>?
          </h2>
          <p className="text-muted-foreground text-lg max-w-xl mx-auto">
            Professional-grade audio enhancement powered by cutting-edge AI technology.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="bg-card border border-border rounded-2xl p-6 hover:border-primary/40 transition-all duration-300 group hover:glow-primary"
            >
              <div className="w-11 h-11 rounded-xl bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                <feature.icon className="w-5 h-5 text-primary" />
              </div>
              <h3 className="text-foreground font-semibold text-lg mb-2">{feature.title}</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;
