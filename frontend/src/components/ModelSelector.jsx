// src/components/ModelSelector.jsx

export default function ModelSelector({
                                          models,
                                          value,
                                          onChange
                                      }) {
    return (
        <select
            value={value}
            onChange={e => onChange(e.target.value)}
            className="model-selector glass"
        >
            {models.map(model => (
                <option key={model} value={model}>
                    {model}
                </option>
            ))}
        </select>
    );
}