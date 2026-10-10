const express = require('express');
const si = require('systeminformation');
const app = express();
const PORT = 3000;

app.get('/metrics', async (req, res) => {
    try {
        const cpu = await si.currentLoad();
        const mem = await si.mem();
        const graphics = await si.graphics();
        
        res.json({
            status: "online",
            timestamp: new Date(),
            cpu_load: cpu.currentLoad.toFixed(2) + "%",
            memory_total_gb: (mem.total / 1024 / 1024 / 1024).toFixed(2),
            memory_used_gb: (mem.used / 1024 / 1024 / 1024).toFixed(2),
            gpu_info: graphics.controllers
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(PORT, () => {
    console.log(`MCP Monitor rodando na porta ${PORT}`);
});
