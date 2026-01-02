import React, { useState, useEffect } from 'react';
import { 
    CssBaseline, Container, Grid, Paper, Typography, Chip, Button, 
    Box, AppBar, Toolbar, CircularProgress, Card, CardMedia, CardContent,
    Tabs, Tab, TextField
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import FaceIcon from '@mui/icons-material/Face';
import SearchIcon from '@mui/icons-material/Search';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import { getAttributes, generateSketch, searchDatabase } from './api';

const theme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#90caf9',
        },
        secondary: {
            main: '#f48fb1',
        },
    },
});

function App() {
    const [attributes, setAttributes] = useState([]);
    const [selectedAttributes, setSelectedAttributes] = useState([]);
    const [generatedImage, setGeneratedImage] = useState(null);
    const [loading, setLoading] = useState(false);
    const [searchResults, setSearchResults] = useState([]);
    const [tabValue, setTabValue] = useState(0);
    const [filterText, setFilterText] = useState('');

    useEffect(() => {
        loadAttributes();
    }, []);

    const loadAttributes = async () => {
        const attrs = await getAttributes();
        setAttributes(attrs);
    };

    const toggleAttribute = (attr) => {
        if (selectedAttributes.includes(attr)) {
            setSelectedAttributes(selectedAttributes.filter(a => a !== attr));
        } else {
            setSelectedAttributes([...selectedAttributes, attr]);
        }
    };

    const handleGenerate = async () => {
        setLoading(true);
        try {
            const imgBase64 = await generateSketch(selectedAttributes);
            setGeneratedImage(`data:image/png;base64,${imgBase64}`);
        } catch (error) {
            alert("Failed to generate sketch");
        }
        setLoading(false);
    };

    const handleSearch = async () => {
        setLoading(true);
        try {
            const results = await searchDatabase(selectedAttributes);
            setSearchResults(results);
        } catch (error) {
            alert("Failed to search database");
        }
        setLoading(false);
    };

    const filteredAttributes = attributes.filter(attr => 
        attr.toLowerCase().includes(filterText.toLowerCase())
    );

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <AppBar position="static">
                <Toolbar>
                    <FaceIcon sx={{ mr: 2 }} />
                    <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                        Sketch Artist AI
                    </Typography>
                </Toolbar>
            </AppBar>

            <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
                <Grid container spacing={3}>
                    {/* Left Panel: Controls */}
                    <Grid item xs={12} md={4}>
                        <Paper sx={{ p: 2, height: '100%' }}>
                            <Typography variant="h6" gutterBottom>
                                Facial Attributes
                            </Typography>
                            <TextField 
                                fullWidth 
                                label="Filter Attributes" 
                                variant="outlined" 
                                size="small" 
                                sx={{ mb: 2 }}
                                value={filterText}
                                onChange={(e) => setFilterText(e.target.value)}
                            />
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                                {filteredAttributes.map(attr => (
                                    <Chip
                                        key={attr}
                                        label={attr.replace(/_/g, ' ')}
                                        onClick={() => toggleAttribute(attr)}
                                        color={selectedAttributes.includes(attr) ? "primary" : "default"}
                                        variant={selectedAttributes.includes(attr) ? "filled" : "outlined"}
                                        clickable
                                    />
                                ))}
                            </Box>
                        </Paper>
                    </Grid>

                    {/* Right Panel: Output */}
                    <Grid item xs={12} md={8}>
                        <Paper sx={{ p: 2, minHeight: '80vh' }}>
                            <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)} centered sx={{ mb: 3 }}>
                                <Tab icon={<AutoFixHighIcon />} label="Generate" />
                                <Tab icon={<SearchIcon />} label="Database Search" />
                            </Tabs>

                            {/* GENERATOR TAB */}
                            {tabValue === 0 && (
                                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                                    <Box 
                                        sx={{ 
                                            width: 512, 
                                            height: 512, 
                                            border: '2px dashed #555', 
                                            borderRadius: 2,
                                            display: 'flex',
                                            justifyContent: 'center',
                                            alignItems: 'center',
                                            overflow: 'hidden',
                                            mb: 3,
                                            bgcolor: '#1e1e1e'
                                        }}
                                    >
                                        {loading ? (
                                            <CircularProgress />
                                        ) : generatedImage ? (
                                            <img src={generatedImage} alt="Generated Sketch" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                                        ) : (
                                            <Typography color="text.secondary">Select attributes and click Generate</Typography>
                                        )}
                                    </Box>

                                    <Button 
                                        variant="contained" 
                                        size="large" 
                                        onClick={handleGenerate}
                                        disabled={loading}
                                        startIcon={<AutoFixHighIcon />}
                                    >
                                        Generate Sketch
                                    </Button>
                                    
                                    <Box sx={{ mt: 2 }}>
                                        <Typography variant="caption" color="text.secondary">
                                            Current Attributes: {selectedAttributes.join(", ") || "None"}
                                        </Typography>
                                    </Box>
                                </Box>
                            )}

                            {/* SEARCH TAB */}
                            {tabValue === 1 && (
                                <Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
                                        <Button 
                                            variant="contained" 
                                            onClick={handleSearch}
                                            disabled={loading}
                                            startIcon={<SearchIcon />}
                                        >
                                            Find Similar Sketches
                                        </Button>
                                    </Box>

                                    <Grid container spacing={2}>
                                        {searchResults.map((result) => (
                                            <Grid item xs={6} sm={4} md={3} key={result.filename}>
                                                <Card>
                                                    {/* Note: We assume backend serves images at /photos/ */}
                                                    <CardMedia
                                                        component="img"
                                                        height="200"
                                                        image={`http://localhost:8000/photos/${result.filename}`} 
                                                        // Fallback logic might be needed if ext differs, but backend is serving from photos dir
                                                        alt={result.filename}
                                                        onError={(e) => {
                                                            // Try jpg if png fails (or vice versa depending on your data)
                                                            if (e.target.src.endsWith('.png')) {
                                                                e.target.src = e.target.src.replace('.png', '.jpg');
                                                            }
                                                        }}
                                                    />
                                                    <CardContent>
                                                        <Typography variant="body2" color="text.secondary">
                                                            Score: {result.score} matches
                                                        </Typography>
                                                        <Typography variant="caption" display="block" sx={{ lineHeight: 1.2, mt: 1 }}>
                                                            {result.attributes.slice(0, 3).join(", ")}...
                                                        </Typography>
                                                    </CardContent>
                                                </Card>
                                            </Grid>
                                        ))}
                                    </Grid>
                                    {searchResults.length === 0 && !loading && (
                                        <Typography align="center" color="text.secondary" sx={{ mt: 5 }}>
                                            No results found. Select attributes and click search.
                                        </Typography>
                                    )}
                                </Box>
                            )}
                        </Paper>
                    </Grid>
                </Grid>
            </Container>
        </ThemeProvider>
    );
}

export default App;