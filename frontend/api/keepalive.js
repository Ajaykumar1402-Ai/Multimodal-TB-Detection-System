export default async function handler(req, res) {
  try {
    // Determine backend URL from environment variables or use the production Render domain fallback
    const BACKEND_URL = process.env.BACKEND_URL || 'https://multimodal-tb-detection-system.onrender.com';
    
    console.log(`[Keep-Alive] Pinging backend health endpoint: ${BACKEND_URL}/health`);
    
    const response = await fetch(`${BACKEND_URL}/health`, {
      method: 'GET',
      headers: {
        'User-Agent': 'Vercel-KeepAlive-Cron'
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP Error Status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('[Keep-Alive] Ping successfully received by backend:', data);
    
    return res.status(200).json({ 
      pinged: true, 
      backend_status: data.status || 'awake',
      model_loaded: data.model_loaded,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('[Keep-Alive] Error during keep-alive ping:', error.message);
    return res.status(500).json({ 
      pinged: false, 
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
}
