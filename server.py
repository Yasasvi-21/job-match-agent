from mcp.server.mcpserver import MCPServer                                 
from day2_loop import fetch_posting as _fetch_posting, get_resume as _get_resume

mcp = MCPServer("job-match")    
                                            
@mcp.tool()
def fetch_posting(company: str, job_id: str) -> str:
    """Fetches one specific job posting from Greenhouse and returns its title, location, and full description as plain text.
    Use this when you need a particular job's requirements.
    company is the Greenhouse board name, e.g. 'stripe' or 'anthropic'.
    job_id is the numeric job id, e.g. '8144196'.
    Returns a message starting with 'Error' if the posting cannot be found."""
    return _fetch_posting(company, job_id)

@mcp.tool()
def get_resume() -> str:
    """Returns the candidate's resume as plain text, including education, experience, skills, and projects.
       Use this whenever you need the candidate's background, for example to judge job fit. Do not use it for job postings."""

    return _get_resume()

if __name__ == "__main__":
    mcp.run()