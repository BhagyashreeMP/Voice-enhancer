import { ArrowRight } from "lucide-react";

const steps = [
  { step: "01", title: "Upload", description: "Drag & drop your audio file or browse from your device." },
  { step: "02", title: "Enhance", description: "Our AI analyzes and enhances your audio in seconds." },
  { step: "03", title: "Download", description: "Get your studio-quality enhanced audio file instantly." },
];

const HowItWorks = () => {
  return (
    <section className="py-20 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-14">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-3">
            How It <span className="text-gradient">Works</span>
          </h2>
          <p className="text-muted-foreground text-lg">Three simple steps to perfect audio.</p>
        </div>

        <div className="flex flex-col md:flex-row items-center gap-6">
          {steps.map((item, i) => (
            <div key={item.step} className="flex items-center gap-6 flex-1">
              <div className="bg-card border border-border rounded-2xl p-6 flex-1 text-center hover:border-primary/40 transition-all duration-300">
                <div className="text-primary font-mono text-sm mb-2">{item.step}</div>
                <h3 className="text-foreground font-semibold text-xl mb-2">{item.title}</h3>
                <p className="text-muted-foreground text-sm">{item.description}</p>
              </div>
              {i < steps.length - 1 && (
                <ArrowRight className="w-5 h-5 text-muted-foreground hidden md:block shrink-0" />
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
