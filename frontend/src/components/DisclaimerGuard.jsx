import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

const DisclaimerGuard = ({ children }) => {
  const navigate = useNavigate();
  const [allowed, setAllowed] = useState(false);

  useEffect(() => {
    const ack = sessionStorage.getItem('tb_consent_ack');
    if (ack !== 'true') {
      // Redirect with replace — removes results page from history stack
      // so Back button cannot reach it
      navigate('/', { replace: true });
    } else {
      setAllowed(true);
    }
  }, [navigate]);

  // Return null until we've confirmed the key exists.
  // This means ZERO frames of results content are ever painted.
  if (!allowed) {
    return null;
  }

  return children;
};

export default DisclaimerGuard;
