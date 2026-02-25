const stats = [
  { value: "50K+", label: "Files Enhanced" },
  { value: "98%", label: "Quality Score" },
  { value: "3s", label: "Avg Processing" },
  { value: "10K+", label: "Happy Users" },
];

const StatsSection = () => {
  return (
    <section className="py-16 px-4 border-y border-border">
      <div className="max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8">
        {stats.map((stat) => (
          <div key={stat.label} className="text-center">
            <div className="text-3xl md:text-4xl font-bold text-gradient mb-1">{stat.value}</div>
            <div className="text-muted-foreground text-sm">{stat.label}</div>
          </div>
        ))}
      </div>
    </section>
  );
};

export default StatsSection;
