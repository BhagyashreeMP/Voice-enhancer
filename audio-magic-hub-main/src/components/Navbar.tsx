import { Waves } from "lucide-react";

const Navbar = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-xl border-b border-border">
      <div className="max-w-5xl mx-auto flex items-center justify-between px-4 h-16">
        <div className="flex items-center gap-2">
          <div className="w-9 h-9 rounded-xl bg-primary/20 flex items-center justify-center">
            <Waves className="w-5 h-5 text-primary" />
          </div>
          <span className="text-foreground font-bold text-lg tracking-tight">Texutra World</span>
        </div>
        <div className="flex items-center gap-6 text-sm">
          <a href="#features" className="text-muted-foreground hover:text-foreground transition-colors hidden sm:block">Features</a>
          <a href="#how-it-works" className="text-muted-foreground hover:text-foreground transition-colors hidden sm:block">How It Works</a>
          <a href="#enhance" className="bg-primary text-primary-foreground px-4 py-2 rounded-lg font-medium hover:bg-primary/90 transition-colors text-sm">
            Try Now
          </a>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
